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
output_file = IOfiles.get('outfile')
print(f"datalist file = {datalist_file}")
print(f"output file = {output_file}")
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
BG_file = (ExpInfo.get('BGfile'))
countTimeBG = float(ExpInfo.get('countTimeBG'))
print(f"Ei = {Ei} meV (k={k_len} A-1)")
print(f"Background file = {BG_file}")
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
axis1 = (SliceInfo.get('Axis1'))
mesh1 = float(SliceInfo.get('Mesh1'))
axis2 = (SliceInfo.get('Axis2'))
mesh2 = float(SliceInfo.get('Mesh2'))
zeroIntFilling = SliceInfo.get('zeroIntFilling')
print(f"Qx_max = {h_max}")
print(f"Qx_min = {h_min}")
print(f"Qy_max = {k_max}")
print(f"Qy_min = {k_min}")
print(f"Qz_max = {l_max}")
print(f"Qz_min = {l_min}")
print(f"Slice plane = {axis1}-{axis2}, ({mesh1} x {mesh2})")
print(f"Zero intensity filling = {zeroIntFilling}")
print("")

print(f"========== UB info ===========")
UBInfo = cfg['UB Info']
u11 = float(UBInfo.get('u11'))
u12 = float(UBInfo.get('u12'))
u13 = float(UBInfo.get('u13'))
u21 = float(UBInfo.get('u21'))
u22 = float(UBInfo.get('u22'))
u23 = float(UBInfo.get('u23'))
u31 = float(UBInfo.get('u31'))
u32 = float(UBInfo.get('u32'))
u33 = float(UBInfo.get('u33'))

UB = np.array([[u11, u12, u13], [u21, u22, u23], [u31, u32, u33]])
UB_inv = np.linalg.inv(UB)
print(f"UB = {UB}")
print(f"UB_inv = {UB_inv}")
# preparation for data slicing ===========

dQ1=0.0
dQ2=0.0

if (axis1=="a"):
    dQ1=(h_max-h_min)/mesh1
elif (axis1=="b"):
    dQ1=(k_max-k_min)/mesh1
elif (axis1=="c"):
    dQ1=(l_max-l_min)/mesh1

if (axis2=="a"):
    dQ2=(h_max-h_min)/mesh2
elif (axis2=="b"):
    dQ2=(k_max-k_min)/mesh2
elif (axis2=="c"):
    dQ2=(l_max-l_min)/mesh2

# reading sensitivity and BG files

pixel_sensitivity=np.zeros((pixelNumX*pixelNumY))
pixel_BG=np.zeros((pixelNumX*pixelNumY))

FHsens=open(sensitivity_file,"r")
for line in FHsens:
    if "#" not in line:
        values = line.split()
        if len(values) == 3:
            Xtemp=int(float(values[0]))
            Ytemp=int(float(values[1]))
            pixel_sensitivity[Xtemp*pixelNumX+Ytemp]=float(values[2])
FHsens.close()

FHBG=open(BG_file,"r")
for line in FHBG:
    if "#" not in line:
        values = line.split()
        if len(values) == 3:
            Xtemp=int(float(values[0]))
            Ytemp=int(float(values[1]))
            pixel_BG[Xtemp*pixelNumX+Ytemp]=float(values[2])
FHBG.close()


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

Intensity=np.zeros((int(mesh1),int(mesh2)))
SqError=np.zeros((int(mesh1),int(mesh2)))
dataNum=np.zeros((int(mesh1),int(mesh2)))

FH1=open(datalist_file,"r")
for line in FH1:
    temp=line.split()
    intMapFile=temp[0]
    C2=float(temp[1])+C2ofst
    countTime=float(temp[2])
    Rot_C2 = np.array( 
        [[ np.cos(np.pi/180.0*(-C2)),  -np.sin(np.pi/180.0*(-C2)),  0 ],
        [ np.sin(np.pi/180.0*(-C2)),  np.cos(np.pi/180.0*(-C2)),  0 ],
        [  0,   0, 1.0 ]])
    Q_C2rot = Q0 @ Rot_C2.T

    FH2=open(intMapFile,"r")
    for line2 in FH2:
        if "#" not in line2:
            values = line2.split()
            if len(values) == 3:
                Xtemp=int(float(values[0]))
                Ytemp=int(float(values[1]))
                idx = Xtemp * pixelNumX + Ytemp

                Qvec = Q_C2rot[idx]
                hkl = Qvec @ UB_inv.T

                h, k, l = hkl
                if (h_min <= h <=h_max) and (k_min <= k <=k_max) and (l_min <= l <= l_max):
                    i=0
                    j=0
                    if (axis1=="a"):
                        i = int(((h-h_min)/dQ1))
                    elif (axis1=="b"):
                        i = int(((k-k_min)/dQ1))
                    elif (axis1=="c"):
                        i = int(((l-l_min)/dQ1))

                    if (axis2=="a"):
                        j = int(((h-h_min)/dQ2))
                    elif (axis2=="b"):
                        j = int(((k-k_min)/dQ2))
                    elif (axis2=="c"):
                        j = int(((l-l_min)/dQ2))

                    Intensity[i][j]+=(float(values[2])/countTime -pixel_BG[Xtemp*pixelNumX+Ytemp]/countTimeBG)/pixel_sensitivity[Xtemp*pixelNumX+Ytemp]
                    SqError[i][j]+=float(values[2])/countTime**2.0/pixel_sensitivity[Xtemp*pixelNumX+Ytemp]**2.0
                    dataNum[i][j]+=1

# step 8: writing sliced data in the output file.

FHR=open(output_file,"w")
FHR.write(f"#hkl{axis1}  hkl{axis2}  Intensity  Error  dataNum\n")
for i in range(int(mesh1)):
    for j in range(int(mesh2)):
        Q1=0.0
        Q2=0.0
        if (axis1 == "a"):
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

        if Intensity[i][j] > 0:
            FHR.write("{0}  {1}  {2}  {3}  {4}\n".format(Q1,Q2,Intensity[i][j],np.sqrt(SqError[i][j]),dataNum[i][j]))
        else:
            FHR.write("{0}  {1}  {2}  {3}  {4}\n".format(Q1,Q2,zeroIntFilling,0,dataNum[i][j]))

    FHR.write("\n")

FHR.close()
