import sys
import numpy as np
import configparser

args=sys.argv
if (len(args)<2):
	print("usage:  python ponta_2dpsd.py [config file] ")
	sys.exit()

cfg = configparser.ConfigParser()
cfg.read(args[1], encoding='utf-8')

print(f"========== I/O files ===========")
IOfiles = cfg['IO Files']
datalist_file = IOfiles.get('datalist')
slice_file = IOfiles.get('slicefile')
cut_file = IOfiles.get('cutfile')
print(f"datalist file = {datalist_file}")
print(f"slice file = {slice_file}")
print(f"1D cut file = {cut_file}")
print("")

print(f"========== Detector info ===========")
DetInfo = cfg['Detector Info']
sensitivity_file = (DetInfo.get('sensFile'))
SDD = float(DetInfo.get('SDD'))
A2Center = float(DetInfo.get('A2Center'))
pixelNumX = int(DetInfo.get('pixelNumX'))
pixelNumY = int(DetInfo.get('pixelNumY'))
pixelSizeX = float(DetInfo.get('pixelSizeX'))
pixelSizeY = float(DetInfo.get('pixelSizeY'))
x0 = float(DetInfo.get('centerPixelX'))
y0 = float(DetInfo.get('centerPixelY'))
alpha = float(DetInfo.get('alpha'))
print(f"Sensitivity file = {sensitivity_file}")
print(f"SDD = {SDD}")
print(f"A2 center = {A2Center}")
print(f"pixelNumX = {pixelNumX}")
print(f"pixelNumY = {pixelNumY}")
print(f"pixelSizeX = {pixelSizeX}")
print(f"pixelSizeY = {pixelSizeY}")
print(f"x0 = {x0}")
print(f"y0 = {y0}")
print(f"alpha = {alpha}")
print("")

print(f"========== Experiment info ===========")
ExpInfo = cfg['Exp Info']
Ei = float(ExpInfo.get('Ei'))
k_len = np.sqrt(Ei/2.072)
print(f"Ei = {Ei} meV (k={k_len} A-1)")
print("")

print(f"========== Sample info ===========")
SampleInfo = cfg['Sample Info']
C2ofst = float(SampleInfo.get('C2ofst'))
print(f"C2 offset = {C2ofst}")
print("")


print(f"========== Slice info ===========")
SliceInfo = cfg['Slice Info']
h_max = float(SliceInfo.get('h_max'))
h_min = float(SliceInfo.get('h_min'))
k_max = float(SliceInfo.get('k_max'))
k_min = float(SliceInfo.get('k_min'))
l_max = float(SliceInfo.get('l_max'))
l_min = float(SliceInfo.get('l_min'))
hk2_max = float(SliceInfo.get('hk2_max'))
hk2_min = float(SliceInfo.get('hk2_min'))
axis1 = SliceInfo.get('Axis1')
mesh1 = float(SliceInfo.get('Mesh1'))
axis2 = SliceInfo.get('Axis2')
mesh2 = float(SliceInfo.get('Mesh2'))
zeroIntFilling = SliceInfo.get('zeroIntFilling')

print(f"========== 1Dcut Info info ===========")
CutInfo = cfg['1Dcut Info']
h_max_cut = float(CutInfo.get('h_max'))
h_min_cut = float(CutInfo.get('h_min'))
k_max_cut = float(CutInfo.get('k_max'))
k_min_cut = float(CutInfo.get('k_min'))
l_max_cut = float(CutInfo.get('l_max'))
l_min_cut = float(CutInfo.get('l_min'))
hk2_max_cut = float(CutInfo.get('hk2_max'))
hk2_min_cut = float(CutInfo.get('hk2_min'))
axis_cut = (CutInfo.get('Axis'))
mesh_cut = float(CutInfo.get('Mesh'))
zeroIntFilling_cut = CutInfo.get('zeroIntFilling')

UBInfo = cfg['UB Info']
UB = np.array([[float(UBInfo.get('u11')), float(UBInfo.get('u12')), float(UBInfo.get('u13'))],
               [float(UBInfo.get('u21')), float(UBInfo.get('u22')), float(UBInfo.get('u23'))],
               [float(UBInfo.get('u31')), float(UBInfo.get('u32')), float(UBInfo.get('u33'))]])
UB_inv = np.linalg.inv(UB)


# preparation for data slicing ===========

dQ1=0.0
dQ2=0.0

if axis1 == "hk2":
    dQ1 = (hk2_max - hk2_min) / mesh1
elif axis1 == "a":
    dQ1 = (h_max - h_min) / mesh1
elif axis1 == "b":
    dQ1 = (k_max - k_min) / mesh1
elif axis1 == "c":
    dQ1 = (l_max - l_min) / mesh1

if axis2 == "a":
    dQ2 = (h_max - h_min) / mesh2
elif axis2 == "b":
    dQ2 = (k_max - k_min) / mesh2
elif axis2 == "c":
    dQ2 = (l_max - l_min) / mesh2

if (axis_cut=="hk2"):
    dQc=(hk2_max_cut-hk2_min_cut)/mesh_cut
elif (axis_cut=="a"):
    dQc=(h_max_cut-h_min_cut)/mesh_cut
elif (axis_cut=="b"):
    dQc=(k_max_cut-k_min_cut)/mesh_cut
elif (axis_cut=="c"):
    dQc=(l_max_cut-l_min_cut)/mesh_cut


# reading sensitivity 

pixel_sensitivity=np.zeros((pixelNumX*pixelNumY))

FHsens=open(sensitivity_file,"r")
for line in FHsens:
    if "#" not in line:
        values = line.split()
        if len(values) == 3:
            Xtemp=int(float(values[0]))
            Ytemp=int(float(values[1]))
            pixel_sensitivity[Xtemp*pixelNumX+Ytemp]=float(values[2])
FHsens.close()



# step 0: preparing a matrix with the size of (pixelNumX*pixelNumY,3)

pixel_positions=np.zeros((pixelNumX*pixelNumY,3))

# step 1: define pixel positions on the yz plane.

for i in range(pixelNumX):
    for j in range(pixelNumY):
        zpos_temp = (float(i)-x0)*pixelSizeX   # Xpixel of the detector -> Z direction
        ypos_temp = (float(j)-y0)*pixelSizeY   # Ypixel of the detector -> Y direction
        pixel_positions[i*pixelNumX+j][0]= 0.0       # kf_array[0]=(Xpic=0,Ypic=0), kf_array[1]=(Xpic=0,Ypic=1), kf_array[2]=(Xpic=0,Ypic=2).....
        pixel_positions[i*pixelNumX+j][1]= ypos_temp
        pixel_positions[i*pixelNumX+j][2]= zpos_temp

# step 2: z-rotation by alpha

Rot_alpha = np.array( 
    [[ np.cos(np.pi/180.0*(alpha)),  -np.sin(np.pi/180.0*(alpha)),  0 ],
     [ np.sin(np.pi/180.0*(alpha)),  np.cos(np.pi/180.0*(alpha)),  0 ],
     [  0,   0, 1.0 ]])

pixel_positions = pixel_positions @ Rot_alpha.T

# step 3: x-translation by SDD

trans_x = np.array([SDD,0,0])

pixel_positions = pixel_positions + trans_x

# step 4: z-rotation by A2Center

Rot_A2 = np.array( 
    [[ np.cos(np.pi/180.0*(A2Center)),  -np.sin(np.pi/180.0*(A2Center)),  0 ],
     [ np.sin(np.pi/180.0*(A2Center)),  np.cos(np.pi/180.0*(A2Center)),  0 ],
     [  0,   0, 1.0 ]])

pixel_positions = pixel_positions @ Rot_A2.T

# step 5: calculate kf vectors from the vectors pointing the pixel positions

kf_array=np.zeros((pixelNumX*pixelNumY,3))

for p in range(len(pixel_positions)):
    kf_array[p]=pixel_positions[p]/np.linalg.norm(pixel_positions[p])*k_len

# step 6: calculate Q0 (Q-vectors at C2=0) by Q=ki-kf

ki = np.array([k_len,0,0])

Q0=ki-kf_array


# step 7: reading observed intensities and map them on the specified plane in the Q space.

Intensity_down=np.zeros((int(mesh1),int(mesh2)))
Intensity_up=np.zeros((int(mesh1),int(mesh2)))
SqError_down=np.zeros((int(mesh1),int(mesh2)))
SqError_up=np.zeros((int(mesh1),int(mesh2)))
dataNum=np.zeros((int(mesh1),int(mesh2)))

Intensity_down_cut=np.zeros((int(mesh_cut)))
Intensity_up_cut=np.zeros((int(mesh_cut)))
SqError_down_cut=np.zeros((int(mesh_cut)))
SqError_up_cut=np.zeros((int(mesh_cut)))
dataNum_cut=np.zeros((int(mesh_cut)))

FH1=open(datalist_file,"r")
for line in FH1:
    if "#" not in line:
        temp=line.split()
        C2=float(temp[0])+C2ofst
        intMapFile_down=temp[1]
        intMapFile_up=temp[2]
        intMapFile_BG=temp[3]
        countTime_down=float(temp[4])
        countTime_up=float(temp[5])
        countTime_BG=float(temp[6])
        Rot_C2 = np.array( 
            [[ np.cos(np.pi/180.0*(-C2)),  -np.sin(np.pi/180.0*(-C2)),  0 ],
            [ np.sin(np.pi/180.0*(-C2)),  np.cos(np.pi/180.0*(-C2)),  0 ],
            [  0,   0, 1.0 ]])
        Q_C2rot = Q0 @ Rot_C2.T

        pixel_BG=np.zeros((pixelNumX*pixelNumY))
        FHBG=open(intMapFile_BG,"r")
        for line in FHBG:
            if ("#" not in line) and ("X" not in line):
                values = line.split()
                if len(values) == 3:
                    Xtemp=int(float(values[0]))
                    Ytemp=int(float(values[1]))
                    pixel_BG[Xtemp*pixelNumX+Ytemp]=float(values[2])
        FHBG.close()

        Xpos=np.zeros(pixelNumX*pixelNumY)
        Ypos=np.zeros(pixelNumX*pixelNumY)
        IntMap_down=np.zeros(pixelNumX*pixelNumY)
        IntMap_up=np.zeros(pixelNumX*pixelNumY)

        FH_down=open(intMapFile_down,"r")
        line_count=0
        for line_down in FH_down:
            if ("#" not in line_down) and ("X" not in line_down):
                values = line_down.split()
                if len(values) == 3:
                    Xpos[line_count]=int(float(values[0]))
                    Ypos[line_count]=int(float(values[1]))
                    IntMap_down[line_count]=float(values[2])
                    line_count+=1
        FH_down.close()

        FH_up=open(intMapFile_up,"r")
        line_count=0
        for line_up in FH_up:
            if ("#" not in line_up) and ("X" not in line_up):
                values = line_up.split()
                if len(values) == 3:
                    # we assume that the orders of Xpos and Ypos are the same as those in intMapFile_down.
                    IntMap_up[line_count]=float(values[2])
                    line_count+=1
        FH_up.close()

        for k in range(len(IntMap_down)):
            pixelIndex=int(Xpos[k]*pixelNumX+Ypos[k])
            #Qx=Q_C2rot[pixelIndex][0]
            #Qy=Q_C2rot[pixelIndex][1]
            #Qz=Q_C2rot[pixelIndex][2]
            Qvec = Q_C2rot[pixelIndex]
            hkl = Qvec @ UB_inv.T
            h, k, l = hkl

            if (h_min <= h <=h_max) and (k_min <= k <=k_max) and (l_min <= l <=l_max):
                i=0
                j=0
                if (axis1=="hk2"):
                    x = h + k / 2
                    i = int(((x-hk2_min)/dQ1))
                elif (axis1=="a"):
                    i = int(((h-h_min)/dQ1))
                elif (axis1=="b"):
                    #y = -k
                    i = int(((k-k_min)/dQ1))
                elif (axis1=="c"):
                    i = int(((l-l_min)/dQ1))

                if (axis2=="a"):
                    j = int(((h-h_min)/dQ2))
                elif (axis2=="b"):
                    #exp160,170においてa*がa*、-a*かの整合性をとるため追加
                    y = -k
                    #j = int(((k-k_min)/dQ2))
                    j = int(((y-k_min)/dQ2))
                elif (axis2=="c"):
                    j = int(((l-l_min)/dQ2))

                Intensity_down[i][j]+=(IntMap_down[pixelIndex]/countTime_down - pixel_BG[pixelIndex]/countTime_BG)/pixel_sensitivity[pixelIndex]
                Intensity_up[i][j]+=(IntMap_up[pixelIndex]/countTime_up - pixel_BG[pixelIndex]/countTime_BG)/pixel_sensitivity[pixelIndex]
                SqError_down[i][j]+=(IntMap_down[pixelIndex]/countTime_down**2.0 + pixel_BG[pixelIndex]/countTime_BG**2.0)/pixel_sensitivity[pixelIndex]**2.0   # error of pixel_sensitivity is not taken into account.
                SqError_up[i][j]+=(IntMap_up[pixelIndex]/countTime_up**2.0 + pixel_BG[pixelIndex]/countTime_BG**2.0)/pixel_sensitivity[pixelIndex]**2.0   # error of pixel_sensitivity is not taken into account.
                dataNum[i][j]+=1

            if (h_min_cut <= h <=h_max_cut) and (k_min_cut <= -k <=k_max_cut) and (l_min_cut <= l <=l_max_cut):
                p=0
                if (axis_cut=="hk2"):
                    p = int(((x-hk2_min_cut)/dQc))
                elif (axis_cut=="a"):
                    p = int(((h-h_min_cut)/dQc))
                elif (axis_cut=="b"):
                    p = int(((-k-k_min_cut)/dQc))
                elif (axis_cut=="c"):
                    p = int(((l-l_min_cut)/dQc))

                Intensity_down_cut[p]+=(IntMap_down[pixelIndex]/countTime_down - pixel_BG[pixelIndex]/countTime_BG)/pixel_sensitivity[pixelIndex]
                Intensity_up_cut[p]+=(IntMap_up[pixelIndex]/countTime_up - pixel_BG[pixelIndex]/countTime_BG)/pixel_sensitivity[pixelIndex]
                SqError_down_cut[p]+=(IntMap_down[pixelIndex]/countTime_down**2.0 + pixel_BG[pixelIndex]/countTime_BG**2.0)/pixel_sensitivity[pixelIndex]**2.0   # error of pixel_sensitivity is not taken into account.
                SqError_up_cut[p]+=(IntMap_up[pixelIndex]/countTime_up**2.0 +  pixel_BG[pixelIndex]/countTime_BG**2.0)/pixel_sensitivity[pixelIndex]**2.0   # error of pixel_sensitivity is not taken into account.
                dataNum_cut[p]+=1


FH1.close()

# step 8: writing sliced data in the output file.

FHS=open(slice_file,"w")
FHS.write(f"#Q{axis1}  Q{axis2}  Int_down  Error  Int_up  Error dataNum\n")
for i in range(int(mesh1)):
    for j in range(int(mesh2)):
        Q1=0.0
        Q2=0.0
        if (axis1 == "hk2"):
            Q1=hk2_min+dQ1*float(i)
        elif (axis1 == "a"):
            Q1=h_min+dQ1*float(i)
        elif (axis1 == "b"):
            Q1=k_min+dQ1*float(i)
        elif (axis1 == "c"):
            Q1=l_min+dQ1*float(i)

        if (axis2 == "a"):
            Q2=h_min+dQ2*float(j)
        elif (axis2 == "b"):
            Q2=k_min+dQ2*float(j)
        elif (axis2 == "c"):
            Q2=l_min+dQ2*float(j)

        FHS.write("{0}  {1}  {2}  {3}  {4}  {5}  {6}\n".format(Q1,Q2,Intensity_down[i][j],np.sqrt(SqError_down[i][j]),Intensity_up[i][j],np.sqrt(SqError_up[i][j]),dataNum[i][j]))

#        if Intensity[i][j] > 0:
#            FHR.write("{0}  {1}  {2}  {3}  {4}\n".format(Q1,Q2,Intensity[i][j],np.sqrt(SqError[i][j]),dataNum[i][j]))
#        else:
#           FHR.write("{0}  {1}  {2}  {3}  {4}\n".format(Q1,Q2,zeroIntFilling,0,dataNum[i][j]))

    FHS.write("\n")
FHS.close()

FHC=open(cut_file,"w")
FHC.write(f"#Q{axis_cut}  Int_down  Error  Int_up  Error dataNum\n")
for p in range(int(mesh_cut)):
    Qc=0.0
    if (axis_cut == "hk2"):
        Qc=hk2_min_cut+dQc*float(p)
    elif (axis_cut == "a"):
        Qc=h_min_cut+dQc*float(p)
    elif (axis_cut == "b"):
        Qc=k_min_cut+dQc*float(p)
    elif (axis_cut == "c"):
        Qc=k_min_cut+dQc*float(p)

    FHC.write("{0}  {1}  {2}  {3}  {4}  {5}\n".format(Qc,Intensity_down_cut[p],np.sqrt(SqError_down_cut[p]),Intensity_up_cut[p],np.sqrt(SqError_up_cut[p]),dataNum_cut[p]))

#        if Intensity[i][j] > 0:
#            FHR.write("{0}  {1}  {2}  {3}  {4}\n".format(Q1,Q2,Intensity[i][j],np.sqrt(SqError[i][j]),dataNum[i][j]))
#        else:
#           FHR.write("{0}  {1}  {2}  {3}  {4}\n".format(Q1,Q2,zeroIntFilling,0,dataNum[i][j]))

FHC.close()
