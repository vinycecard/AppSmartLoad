import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import matplotlib.mlab as mlb
from matplotlib import pylab
from pylab import *
from time import sleep

######################################
Load=50;#######Load in Kilos #########
######################################
t=0

num_samples=fs=duration=0;
num_samples_2=0;
duration_cal=0;
num_samples_cal=0;
data_ang_cal_z=0;
g=ac=0;
data_g=0;
data_gx=0;
data_gy=0;
data_gz=0;
mean_g=0;
mean_gx=0;
mean_gy=0;
mean_gz=0;
gres_cal=0;
gres_cal_SI=0;
data_ac=0;
data_ac_x=0;
data_ac_y=0;
data_ac_z=0;
ang=0;
ang_x=ang_y=ang_z=ang_z_cal=0;
ang_cal_x=0;
ang_cal_y=0
ang_cal_z=0;
data_ang=0;
data_ang_x=data_ang_y=data_ang_z=0;
i=0;
i1=0;
i2=0;
i3=0;
i4=0;
i5=0;
pitch=roll=yaw=0;
pitch_sin=0;
roll_sin=0;
yaw_sin=0;
pitch_cos=0;
roll_cos=0;
yaw_cos=0;
yawMatrix=0;
rollMatrix=0;
pitchMatrix=0;
R=0;
ac_R=0;
ac_R_x=0;
ac_R_y=0;
ac_R_z=0;
ac_transpose=0;
ac_R_transpose=0;
data_ac_R=0;
data_ac_R_x=0;
data_ac_R_y=0;
data_ac_R_z=0;
m=n=0;
data_ang_x_global_parcial=0;
data_ang_y_global_parcial=0;
data_ang_calibrado=0
i_values=i1_values=0;
i_values_cal=0;
integral_sinal=0;
i_int=0;
i_tempo=0;
v_tempo=0;
v_tempo_progress=0;
v_R=0;
data_v_R=0;
data_v_R_x=data_v_R_y=data_v_R_z=0
f_ac_R_x=0;
Pxx_den_ac_R_x=0;

del data_ang_cal_z,ang_z_cal,ang_cal_x,ang_cal_y,ang_cal_z
del g,ac 
del num_samples,num_samples_2,fs,duration
del num_samples_cal,duration_cal
del data_g,data_gx,data_gy,data_gz
del	mean_g,mean_gx,mean_gy,mean_gz
del gres_cal,gres_cal_SI
del data_ac,data_ac_x,data_ac_y,data_ac_z
del ang,ang_x,ang_y,ang_z
del data_ang
del data_ang_x,data_ang_y,data_ang_z
del i,i1,i2,i3,i4,i5
del pitch,roll,yaw
del pitch_sin,roll_sin,yaw_sin
del pitch_cos,roll_cos,yaw_cos
del yawMatrix,rollMatrix,pitchMatrix
del R,ac_R,ac_transpose,data_ac_R,ac_R_transpose
del data_ac_R_x,data_ac_R_y,data_ac_R_z
del ac_R_x,ac_R_y,ac_R_z
del m,n
del i_values,i1_values,i_values_cal
del data_ang_x_global_parcial,data_ang_y_global_parcial
del integral_sinal
del i_int,i_tempo
del v_tempo_progress,v_tempo
del v_R,data_v_R
del data_v_R_x,data_v_R_y,data_v_R_z
del f_ac_R_x,Pxx_den_ac_R_x
###########################################################################################################################	
print('Calibração')
plotar_analise_espectral=0 #1 para fazer análise espectral
plotar_fig_4=0 #1 para plotar
plotar_fig_5=0 #1 para plotar
plotar_fig_6=1 #1 para plotar
plotar_fig_7=1 #1 para plotar
plotar_fig_8=1 #1 para plotar
plotar_fig_9=1 #1 para plotar

fs=250;
duration_cal=2;
duration=10
tempo_corte=2
amostra_corte=int(fs*tempo_corte)
#print('amostra_corte = ',amostra_corte)
passar_stop_band=0;#0 para ñ e 1 para Sim
num_samples_cal = (fs*duration_cal)+1
num_samples = (fs*duration)+1
# TODO trocar motion pela classe que pega as informações de gravidade/angulos do celeular
#motion.start_updates()
sleep(0.5)
print('Gravando calibração estática...')
data_ang_cal_z=[]
data_g=[]
data_gx=[]
data_gy=[]
data_gz=[]
data_ac=[]
data_ac_x=data_ac_y=data_ac_z=[]
ang=[0,0,0]
ang_x=ang_y=ang_z=[] 
data_ang=[]
data_ang_x=[]
data_ang_y=[]
data_ang_z=[]
data_ang_x_global_parcial=[]
data_ang_y_global_parcial=[]
yawMatrix=rollMatrix=pitchMatrix=[]
R=[]
ac_R=[]
ac_R_x=[]
ac_R_y=[]
ac_R_z=[]
ac_transpose=[]
ac_R_transpose=[]
data_ac_R=[]
data_ac_R_x=[]
data_ac_R_y=[]
data_ac_R_z=[]
i_values=[]
i1_values=[]
i_values_cal=[]
i1=[]
i2=[]
integral_sinal=[]
i_tempo=[]
v_tempo_progress=[]
v_tempo=[]
data_v_R=[]
data_v_R_x=[]
data_v_R_y=[]
data_v_R_z=[]
f_ac_R_x=[]
Pxx_den_ac_R_x=[]

for i in range(num_samples_cal):
	sleep(1/fs);
	ang_cal_z = 0;
	g = [10,10,10];
	#ang_cal_x,ang_cal_y,ang_cal_z=motion.get_attitude();
	#g = motion.get_gravity();
	data_ang_cal_z.append(ang_cal_z);
	gx,gy,gz=g;
	data_g.append(g);
	data_gx.append(gx);
	data_gy.append(gy);
	data_gz.append(gz);

#motion.stop_updates()

i_values_cal = [i/fs for i in 
range(num_samples_cal)]
duration_cal=i_values_cal[-1];
#print('duration_cal = ',duration_cal)
sz_i_values_cal=np.size(i_values_cal)
#print('sz_i_values_cal = ',sz_i_values_cal)
ang_z_cal=np.trapz(data_ang_cal_z,i_values_cal)/duration_cal

sz_data_gx=np.size(data_gx)
#print('sz_data_gx = ',sz_data_gx)
mean_gx=np.trapz(data_gx,i_values_cal)/duration_cal
mean_gy=np.trapz(data_gy,i_values_cal)/duration_cal
mean_gz=np.trapz(data_gz,i_values_cal)/duration_cal
gres_cal=(mean_gx**2+mean_gy**2+mean_gz**2)**(0.5)
gres_cal_SI=gres_cal*9.80665
#print('ang_z_cal = ',ang_z_cal)
#print('gres_cal = ',gres_cal)
print('gres_cal_SI = ',gres_cal_SI)

###################################
print('Motion Plot: accelerometer (motion) data will be recorded for 10 seconds.')
#motion.start_updates()
sleep(2)
print('Gravando dados do movimento...')
# TODO Play the BIP: #######################

for i1 in range(num_samples):
	#if i==amostra_inicio_real:
		#print('Início da Gravação...')
	sleep(1/fs)
	ac = [0,0,0];
	#ac=motion.get_user_acceleration();
	#ang=motion.get_attitude();
	
	ang_x=ang[1]  # Pitch = + X
	ang_y=ang[0];  # Pitch = + Y
	ang_z=ang[2]-ang_z_cal;# Pitch = + Z
	data_ang.append(ang);
	data_ang_x.append(ang_x);
	data_ang_y.append(ang_y);
	data_ang_z.append(ang_z);
	#a=motion.get_gravity()
	#motion.get_attitude()
	#motion.get_magnetic_field()
	
	acx,acy,acz=ac;
	data_ac.append(ac);
	data_ac_x.append(ac);
	data_ac_y.append(ac);
	data_ac_z.append(ac);
	
	alpha=ang_x#*3.1415926
	beta=ang_y#*3.1415926
	gama=ang_z#*3.1415926
	gama_global_real=ang[2]
	 
	alpha_sin=np.sin(alpha)
	alpha_cos=np.cos(alpha)
	beta_sin=np.sin(beta)
	beta_cos=np.cos(beta)
	gama_sin=np.sin(gama)
	gama_cos=np.cos(gama)
	
	#gama_sin_global_real=np.sin(gama_global_real)
	#gama_cos_global_real=np.cos(gama_global_real)
	
	Z_Matrix = np.matrix([[gama_cos, -gama_sin, 0],[gama_sin, gama_cos, 0],[0, 0, 1]])
	
	Y_Matrix = np.matrix([[beta_cos, 0, beta_sin],[0, 1, 0],[-beta_sin, 0, beta_cos]])
	
	X_Matrix = np.matrix([[1, 0, 0],[0, alpha_cos, -alpha_sin],[0, alpha_sin, alpha_cos]])
	R = Z_Matrix * Y_Matrix *X_Matrix
	R=np.matrix(R)
	#R_global_parcial_G=np.matrix([[gama_cos_global_real,gama_sin_global_real,0],[gama_sin_global_real,gama_cos_global_real,0],[0,0,1]])
	#size_R_Global_parcial_G=np.shape(R_global_parcial_G)
	#print('size_R_Global_parcial_G = ',size_R_Global_parcial_G)
	#vetor_ang_xyz=np.matrix([ang_x,ang_y,ang[2]])
	#size_vetor_ang_xyz=np.shape(vetor_ang_xyz)
	#print('size_vetor_ang_xyz = ',size_vetor_ang_xyz)
	#ang_global_parcial=vetor_ang_xyz*R_global_parcial_G;
	#data_ang_x_global_parcial.append(ang_global_parcial[0,0])
	
	#data_ang_y_global_parcial.append(ang_global_parcial[0,1])
	
	#ac=np.matrix(ac)
	#ac_transpose=np.transpose(ac)
	ac_transpose=np.matrix([[acx],[acy],[acz]])
	ac_R=R*ac_transpose
	ac_R_transpose=np.transpose(ac_R)
	data_ac_R.append(ac_R)
	
	ac_R_x,ac_R_y,ac_R_z=ac_R
	
	data_ac_R_x.append(ac_R_x);
	data_ac_R_y.append(ac_R_y);
	data_ac_R_z.append(ac_R_z);

#motion.stop_updates()
print('Capture finished, plotting...')
print('############################')
#print('max_ac_z = ',max(data_ac_R_z))
#print('############################')
########################################
for i1 in range(num_samples-1):
		if data_ang_z[i1+1]>data_ang_z[i1]+3:
			data_ang_z[i1+1]=data_ang_z[i1+1]-2*np.pi
		elif data_ang_z[i1+1]<data_ang_z[i1]-3:
			data_ang_z[i1+1]=data_ang_z[i1+1]+2*np.pi
###### %%%%%%% #######
def filtfilt_sarmet(x,y,fc_1,fc_2,reg,Lv_reg,Exp_reg):
	''' Definir as FCs '''
	if (fc_1==0 and fc_2==10):
		a=np.matrix([1,-1.143,0.4128])
		b=np.matrix([0.0675,0.1349,0.0675])
	elif (fc_1==0 and fc_2==8):
		a=np.matrix([1,-1.3073,0.4918])
		b=np.matrix([0.0461,0.0923,0.0461])
	elif (fc_1==0 and fc_2==6):
		a=np.matrix([1,-1.4755,0.5869])
		b=np.matrix([0.0279,0.0557,0.0279])
	elif (fc_1==0 and fc_2==4):
		a=np.matrix([1,-1.6475,0.7009])
		b=np.matrix([0.0134,0.0267,0.0134])
	elif (fc_1==0.5 and fc_2==1):
		a=np.matrix([1,-3.9517,5.8600,-3.8648,0.9565])
		b=np.matrix([0.9780,-3.9083,5.8606,-3.9083,0.9780])
	elif (fc_1==0.25 and fc_2==0.50):
		a=np.matrix([1,-3.9768,5.9317,-3.9329,0.9780])
		b=np.matrix([0.9890,-3.9548,5.9318,-3.9548,0.9890])
	elif (fc_1==0.025 and fc_2==0.05):
		a=np.matrix([1,-3.9978,5.9933,-3.9933,0.9978])
		b=np.matrix([0.9989,-3.9957,5.9935,-3.9957,0.9989])
	
	''' Finalizar as FCs '''
	vx=[]
	vy=[]
	yfilt=[]
	mb=[]
	ma=[]
	my=[]
	myfilt=[]
	yfiltfilt=[]
	a0=0
	
	
	Lx=np.size(x)
	Lb=np.size(b)
	La=np.size(a)
	#zerolag=0;
	yfilt=np.zeros(Lx)
	N=2;
	N2=np.arange(0,N,1)
		
	#Lv_reg=20
	vx=vy=0
	del vx,vy
	vx=x[0:Lv_reg]
	vy=y[0:Lv_reg]
	p=np.polyfit(vx,vy,Exp_reg)
	reg_vx=np.polyval(p,vx)
		
	for zerolag in N2:
		if zerolag==1:
			#ni=0
			y=np.array(list(reversed(yfilt)))
			#yfilt_a=yfilt
		for ni in np.arange(0,La,1):
			if np.logical_and(zerolag==0,reg==1):
				yfilt[0]=reg_vx[0]
			else:
				yfilt[ni]=np.matrix(y[ni])
			n=La;
			x2=np.arange(La,Lx,1)
			
			for n in x2:
				mb=np.matrix(b[0,np.arange(0,min(Lb,n+1),1)])
				my=np.matrix(y[np.arange(n,max(-1,n-Lb),-1)])
				my=np.transpose(my)
				ma=np.matrix(-a[0,np.arange(1,min(La,n+1),1)])
				myfilt=np.matrix(yfilt[np.arange(n-1,max(-1,n-La),-1)])
				myfilt=np.transpose(myfilt)
				# TODO descobrir pq está dando erro
				#yfilt[n]=[mb*my+ma*myfilt]/a[0,0]
				yfilt[0];
						
	yfiltfilt=list(reversed(yfilt))
	#fig_1a=plt.figure()
	#plt.plot(vx,vy,'k')
	#plt.plot(vx,reg_vx,'r')
	#plt.ylabel('yfilt')
	#plt.show()		
	return yfiltfilt
#########################################
#########################################
i1_values = [i1/fs for i1 in range(num_samples)]
len_i1_values=len(i1_values)
#print('size_i1_values = ',np.shape(i1_values))
########################################

x = np.array(i1_values)
#print('size_x = ',np.shape(x))
y1=np.array(data_ang_x)
#print('size_y1 = ',np.shape(y1))
#########################################
fc_1=0
fc_2=6
reg=3
Lv_reg=50
Exp_reg=0
#########################################
data_ang_x=filtfilt_sarmet(x,y1,fc_1,fc_2,reg,Lv_reg,Exp_reg)
#print('data_ang_x foi filtrado')
#########################################
y2=np.array(data_ang_y)
#########################################
data_ang_y=filtfilt_sarmet(x,y2,fc_1,fc_2,reg,Lv_reg,Exp_reg)
#print('data_ang_y foi filtrado')
#########################################
#########################################
y3=np.array(data_ang_z)
#########################################
data_ang_z=filtfilt_sarmet(x,y3,fc_1,fc_2,reg,Lv_reg,Exp_reg)
#print('data_ang_z foi filtrado')
#########################################
''' Passar Rejeitas Faixa '''
#########################################
fc_1=0.25
fc_2=0.50
reg=3
Lv_reg=50
Exp_reg=0
#########################################
#y1=np.array(data_ang_x)
#########################################
#data_ang_x=filtfilt_sarmet(x,y1,fc_1,fc_2,reg,Lv_reg,Exp_reg)
#########################################
#y2=np.array(data_ang_y)
#########################################
#data_ang_y=filtfilt_sarmet(x,y2,fc_1,fc_2,reg,Lv_reg,Exp_reg)
#########################################
#########################################
#y3=np.array(data_ang_z)
#########################################
#data_ang_z=filtfilt_sarmet(x,y3,fc_1,fc_2,reg,Lv_reg,Exp_reg)
#########################################
#fig_1=plt.figure()
	
#plt.plot(x,y1,'k')
#plt.plot(x,data_ang_x,'r')
#plt.plot(x,y2,'k')
#plt.plot(x,data_ang_y,'b')
#plt.plot(x,y3,'k')
#plt.plot(x,data_ang_z,'g')
#plt.ylabel('yfilt')
#plt.ylabel('ang Raw')
#plt.show()
########################################
data_ang_x=np.transpose(np.matrix(data_ang_x))
size_data_ang_x=np.shape(data_ang_x)
#print('size_data_ang_x = ',size_data_ang_x)
data_ang_y=np.transpose(np.matrix(data_ang_y))
size_data_ang_y=np.shape(data_ang_y)
#print('size_data_ang_y = ',size_data_ang_y)
data_ang_z=np.transpose(np.matrix(data_ang_z))
size_data_ang_z=np.shape(data_ang_z)
#print('size_data_ang_z = ',size_data_ang_z)
data_ang_calibrado=np.hstack((data_ang_x,data_ang_y,data_ang_z))
size_data_ang_calibrado=np.shape(data_ang_calibrado)
#print('size_data_ang_calibrado = ',size_data_ang_calibrado)
data_ac_R_raw=np.squeeze(data_ac_R,axis=2)
data_ac_R_raw=data_ac_R_raw*gres_cal_SI

#print('ac =',ac)
#print('ac_transpose =',ac_transpose)
#print('R = ',R)
#print('ac_R = ',ac_R)
size_ac=np.shape(ac)
#print('size_ac = ',size_ac)
size_ac_transpose=np.shape(ac_transpose)
#print('size_ac_transpose = ',size_ac_transpose)
size_ac_R=np.shape(ac_R)
#print('size_ac_R = ',size_ac_R)
size_ac_R_transpose=np.shape(ac_R_transpose)
#print('size_ac_R_transpose = ',size_ac_R_transpose)
m=np.shape(data_ac_R_raw)
#print('size_data_ac_R_raw = ',m)

data_ang_calibrado_2=data_ang_calibrado.tolist()
#print('data_ang_calibrado_2 = ',type(data_ang_calibrado_2))

#f_ac_R_x, Pxx_den_ac_R_x = signal.periodogram(data_ac_R_x, fs)
#plt.semilogy(f_ac_R_x, Pxx_den_ac_R_x)
#plt.xlabel('frequency [Hz]')
#plt.ylabel('PSD [ac**2/Hz]')
#plt.show()
#print('size_data_ac_R_squeeze = ',np.shape(data_ac_R_x))

#print('data_ac_R_x no SI')
#print('size_data_ac_R_SI = ',np.shape(data_ac_R_x))

#print('############################')
#print('max_ac_z = ',max(data_ac_R_z))
#print('############################')

data_ac_R_x=np.squeeze(data_ac_R_x,axis=1)
data_ac_R_y=np.squeeze(data_ac_R_y,axis=1)
data_ac_R_z=np.squeeze(data_ac_R_z,axis=1)
#print('############################')
data_ac_R_x=data_ac_R_x*gres_cal_SI
data_ac_R_y=data_ac_R_y*gres_cal_SI
data_ac_R_z=data_ac_R_z*gres_cal_SI
#print('max_ac_z = ',max(data_ac_R_z))
#print('############################')

if plotar_analise_espectral==1:
	
	fig_2=plt.figure()
	plt.subplot(3,1,1)
	plt.psd(data_ac_R_x,256,fs)
	plt.ylabel('Acx - PSD (dB/Hz)')

	plt.subplot(3,1,2)
	plt.psd(data_ac_R_y,256,fs)
	plt.ylabel('Acy - PSD (dB/Hz)')

	plt.subplot(3,1,3)
	plt.psd(data_ac_R_z,256,fs)
	plt.ylabel('Acz - PSD (dB/Hz)')
	plt.show()
#########################################
fc_1=0
fc_2=6
reg=3
Lv_reg=50
Exp_reg=0
########################################
y1=np.array(data_ac_R_x)
#########################################
data_ac_R_x=filtfilt_sarmet(x,y1,fc_1,fc_2,reg,Lv_reg,Exp_reg)
#########################################
y2=np.array(data_ac_R_y)
#########################################
data_ac_R_y=filtfilt_sarmet(x,y2,fc_1,fc_2,reg,Lv_reg,Exp_reg)
#########################################
#########################################
y3=np.array(data_ac_R_z)
#########################################
data_ac_R_z=filtfilt_sarmet(x,y3,fc_1,fc_2,reg,Lv_reg,Exp_reg)

#print('############################')
#print('max_ac_z = ',max(data_ac_R_z))
#print('############################')
#########################################
'''       Passar Rejeitas Faixa       '''
#########################################
if passar_stop_band==1:
	fc_1=0.25
	fc_2=0.5
	reg=3
	Lv_reg=50
	Exp_reg=0
	#######################################
	y1=np.array(data_ac_R_x)
	#######################################
	data_ac_R_x=filtfilt_sarmet(x,y1,fc_1,fc_2,reg,Lv_reg,Exp_reg)
	#######################################
	y2=np.array(data_ac_R_y)
	#######################################
	data_ac_R_y=filtfilt_sarmet(x,y2,fc_1,fc_2,reg,Lv_reg,Exp_reg)
	#######################################
	#######################################
	y3=np.array(data_ac_R_z)
	#######################################
	data_ac_R_z=filtfilt_sarmet(x,y3,fc_1,fc_2,reg,Lv_reg,Exp_reg)
	#######################################

#########################################
############ Início do corte ############
#########################################

#print('amostra_corte',amostra_corte)
num_samples_2=num_samples-amostra_corte
#########################################
i1_values_pre_corte = [i1/fs for i1 in range(num_samples)]
len_i1_values_pre_corte=len(i1_values_pre_corte)
#print('size_i1_values_pre_corte = ',np.shape(i1_values_pre_corte))
#########################################
i1_values = [i1/fs for i1 in range(num_samples_2)]
len_i1_values=len(i1_values)
#print('size_i1_values = ',np.shape(i1_values))
########################################
#i1_values=np.array(i1_values)
#i1_values=np.array(i1_values[np.arange(amostra_corte,len_i1_values,1)])
#print('size_i1_values',np.shape(i1_values))
#print('i1_values(0) = ',i1_values[0])
########################################
#data_ac_R=data_ac_R[np.arange(0,amostra_corte,1)]
data_ac_R_x=np.array(data_ac_R_x)
data_ac_R_x=np.array(data_ac_R_x[np.arange(amostra_corte,len_i1_values_pre_corte,1)])
#data_ac_R_x=np.squeeze(data_ac_R_x,axis=2)
#print('size_data_ac_R_x = ',np.shape(data_ac_R_x))

data_ac_R_y=np.array(data_ac_R_y)
data_ac_R_y=np.array(data_ac_R_y[np.arange(amostra_corte,len_i1_values_pre_corte,1)])
#data_ac_R_y=np.squeeze(data_ac_R_y,axis=2)
#print('size_data_ac_R_y = ',np.shape(data_ac_R_y))

data_ac_R_z=np.array(data_ac_R_z)
data_ac_R_z=np.array(data_ac_R_z[np.arange(amostra_corte,len_i1_values_pre_corte,1)])
#data_ac_R_z=np.squeeze(data_ac_R_z,axis=2)
#print('size_data_ac_R_z = ',np.shape(data_ac_R_z))

#data_ang=data_ang[np.arange(0,amostra_corte,1)]
#data_ang_x=np.transpose(np.matrix(data_ang_x))
data_ang_x=np.array(data_ang_x[np.arange(amostra_corte,len_i1_values_pre_corte,1)])
data_ang_x=np.squeeze(data_ang_x,axis=1)
#print('size_data_ang_x = ',np.shape(data_ang_x))

#data_ang_y=np.transpose(np.matrix(data_ang_y))
data_ang_y=np.array(data_ang_y[np.arange(amostra_corte,len_i1_values_pre_corte,1)])
data_ang_y=np.squeeze(data_ang_y,axis=1)
#print('size_data_ang_y = ',np.shape(data_ang_y))

#print('size_data_ang_za = ',np.shape(data_ang_z))
#data_ang_z=np.transpose(np.matrix(data_ang_z))
data_ang_z=np.array(data_ang_z[np.arange(amostra_corte,len_i1_values_pre_corte,1)])
data_ang_z=np.squeeze(data_ang_z,axis=1)
#print('size_data_ang_z = ',np.shape(data_ang_z))

#########################################
############ Início do corte ############
#########################################

#fig_3=plt.figure()
	
#plt.plot(x,y1,'k')
#plt.plot(x,data_ac_R_x,'r')
#plt.plot(x,y2,'r')
#plt.plot(x,data_ac_R_y,'b')
#plt.plot(x,y3,'k')
#plt.plot(x,data_ac_R_z,'g')
#plt.ylabel('ac filt')
#plt.ylabel('ac raw')
#plt.show()
########################################
data_ac_R_x1=np.transpose(np.matrix(data_ac_R_x))
data_ac_R_y1=np.transpose(np.matrix(data_ac_R_y))
data_ac_R_z1=np.transpose(np.matrix(data_ac_R_z))
#size_data_ac_R_x=np.shape(data_ac_R_x)
#print('size_data_ac_R_x = ',size_data_ac_R_x)
########################################
data_ac_R=np.hstack((data_ac_R_x1,data_ac_R_y1,data_ac_R_z1))

data_ac_R_plot=data_ac_R.tolist()

size_data_ac_R=np.shape(data_ac_R)
#print('size_data_ac_R = ',size_data_ac_R)
########################################
data_ang_x=np.transpose(np.matrix(data_ang_x))
data_ang_y=np.transpose(np.matrix(data_ang_y))
data_ang_z=np.transpose(np.matrix(data_ang_z))
data_ang_calibrado=np.hstack((data_ang_x,data_ang_y,data_ang_z))
data_ang_calibrado=data_ang_calibrado.tolist()
########################################

if plotar_fig_4==1:
	fig_4 = plt.figure()
	plt.subplot(2,2,1)
	i1_values_cal = [i1_cal/fs for i1_cal in range(num_samples_cal)]
	for i1_cal, color, label in zip(range(3), 'rgb', 'XYZ'):
		plt.plot(i1_values_cal, [g[i1_cal] for g in data_g], color, label=label, lw=2)
	plt.grid(True)
	plt.xlabel('t')
	plt.ylabel('G')
	plt.gca().set_ylim([-1.0, 1.0])
	plt.legend()
	#plt.show()
		
	plt.subplot(2,2,2)
	#i1_values = [i1/fs for i1 in range(num_samples)]
	for i1, color, label in zip(range(3), 'rgb', 'XYZ'):
		plt.plot(i1_values_pre_corte, [ac[i1] for ac in data_ac], color, label=label, lw=2)
	plt.grid(True)
	plt.xlabel('t')
	plt.ylabel('ac')
	plt.gca().set_ylim([np.min(data_ac),np.max(data_ac)])
	#plt.gca().set_ylim([-1.0, 1.0])
	plt.legend()
	#plt.show()
	
	plt.subplot(2,2,3)
	for i4, color, label in zip(range(3), 'rgb', 'XYZ'):
		plt.plot(i1_values, [ang[i4] for ang in data_ang_calibrado], color, label=label, lw=2)
	plt.grid(True)
	plt.xlabel('t')
	plt.ylabel('ang')
	#ang_min_legend=-2*np.pi
	#ang_max_legend=2*np.pi
	plt.gca().set_ylim([np.min(data_ang_calibrado),np.max(data_ang_calibrado)])
	plt.legend()
	#plt.show()
	
	plt.subplot(2,2,4)
	for i5, color, label in zip(range(3), 'rgb', 'XYZ'):
		plt.plot(i1_values, [ac_R_filt[i5]
	for ac_R_filt in data_ac_R_plot], color, label=label, lw=2)
	plt.grid(True)
	plt.xlabel('t')
	plt.ylabel('ac_R')
	plt.gca().set_ylim([np.min(data_ac_R_plot),np.max(data_ac_R_plot)])
	plt.legend()
	plt.show()
########################################
duration_coleta=duration;
#print('duration_coleta = ',duration_coleta)
sz_i1_values=np.size(i1_values)
#print('sz_i1_values = ',sz_i1_values)
	
#pitch, roll, yaw = [x for x in data_ang]
#pitch=-pitch;
#yaw=-yaw;

#print('g = ',g)
#print('ac = ',ac)
#print('mean_gx = ',mean_gx)
#print('mean_gy = ',mean_gy)
#print('mean_gz = ',mean_gz)

def integra(sinal,v_tempo):
	N=len(v_tempo)
	integral_sinal=np.zeros(N)
	for i in range(N):
		integral_sinal[i]=np.trapz(sinal[0:i+1],v_tempo[0:i+1])
		
	return integral_sinal
#data_ac_R_x=np.squeeze(data_ac_R_x,axis=2)
#data_ac_R_y=np.squeeze(data_ac_R_y,axis=2)
#data_ac_R_z=np.squeeze(data_ac_R_z,axis=2)

#len_data_ac_R_x=np.shape(data_ac_R_x)	
#print('len_data_ac_R_x = ',len_data_ac_R_x)
#len_data_ac_R_y=np.shape(data_ac_R_y)	
#print('len_data_ac_R_y = ',len_data_ac_R_y)
#len_data_ac_R_z=np.shape(data_ac_R_z)	
#print('len_data_ac_R_z = ',len_data_ac_R_z)
#len_i1_values=np.shape(i1_values)	
#print('i1_values = ',len_i1_values)
#########################################
data_v_R_x=integra(data_ac_R_x,i1_values)
#print('data_v_R_x = ',np.shape(data_v_R_x))
data_v_R_y=integra(data_ac_R_y,i1_values)
#print('data_v_R_y = ',np.shape(data_v_R_y))
data_v_R_z=integra(data_ac_R_z,i1_values)
#print('data_v_R_z = ',np.shape(data_v_R_z))
data_v_R=np.transpose(np.vstack((data_v_R_x,data_v_R_y,data_v_R_z)))
#v_R_x=integral_sinal
#print('data_v_R = ',np.shape(data_v_R))

################ Detrend ###############
data_v_R_x_det=mlb.detrend_linear(data_v_R_x)
data_v_R_y_det=mlb.detrend_linear(data_v_R_y)
data_v_R_z_det=mlb.detrend_linear(data_v_R_z)
data_v_R_det=np.transpose(np.vstack((data_v_R_x_det,data_v_R_y_det,data_v_R_z_det)))

#data_Vt=data_v_R_det.tolist()
data_Vx=np.array(data_v_R_x_det)
data_Vy=np.array(data_v_R_y_det)
data_Vz=np.array(data_v_R_z_det)
data_VR=np.sqrt(data_Vx*data_Vx+data_Vy*data_Vy+data_Vz*data_Vz)

data_Vta=np.transpose(np.vstack((data_Vx,data_Vy,data_Vz,data_VR)));
#data_Vta=np.transpose(np.vstack((data_Vx,data_Vy,data_Vz)));
data_Vt=data_Vta.tolist()
#########################################
if plotar_fig_5==1:

	fig_5 = plt.figure()
	plt.subplot(2,1,1)
	#plt.plot(i1_values,v_R_x)
	for i1, color, label in zip(range(3), 'rgb', 'XYZ'):
		plt.plot(i1_values, [v_R[i1] for v_R in data_v_R], color, label=label, lw=2)
	plt.grid(True)
	plt.xlabel('t(s)')
	plt.ylabel('Vx (m/s)')
	plt.gca().set_ylim([data_v_R.min()*1.05,data_v_R.max()*1.05])
	plt.legend()
	#plt.show()
	
	plt.subplot(2,1,2)
	#plt.plot(i1_values,v_R_x)
	for i1, color, label in zip(range(3), 'rgb', 'XYZ'):
		plt.plot(i1_values, [v_R_det[i1] for v_R_det in data_v_R_det], color, label=label, lw=2)
	plt.grid(True)
	plt.xlabel('t(s)')
	plt.ylabel('Vx detrended (m/s)')
	plt.gca().set_ylim([data_v_R_det.min()*1.05,data_v_R_det.max()*1.05])
	plt.legend()
	plt.show()

data_Fx=data_ac_R_x1*Load
data_Fy=data_ac_R_y1*Load
data_Fz=(data_ac_R_z1+gres_cal_SI)*Load

data_Fx=np.array(data_Fx)
data_Fy=np.array(data_Fy)
data_Fz=np.array(data_Fz)
data_FR=np.sqrt(data_Fx*data_Fx+data_Fy*data_Fy+data_Fz*data_Fz)

data_Fta=np.hstack((data_Fx,data_Fy,data_Fz,data_FR));
#data_Fta=np.hstack((data_Fx,data_Fy,data_Fz));
data_Ft=data_Fta.tolist()


np.squeeze(data_ang_x,axis=1)

data_Px=np.transpose(np.matrix(np.multiply(np.squeeze(data_Fx.tolist(),axis=1),data_v_R_x_det.tolist())));
data_Py=np.transpose(np.matrix(np.multiply(np.squeeze(data_Fy.tolist(),axis=1),data_v_R_y_det.tolist())));
data_Pz=np.transpose(np.matrix(np.multiply(np.squeeze(data_Fz.tolist(),axis=1),data_v_R_z_det.tolist())));

data_Px=np.array(data_Px)
data_Py=np.array(data_Py)
data_Pz=np.array(data_Pz)
data_PR=np.sqrt(data_Px*data_Px+data_Py*data_Py+data_Pz*data_Pz)

data_Pta=np.hstack((data_Px,data_Py,data_Pz,data_PR));
#data_Pta=np.hstack((data_Px,data_Py,data_Pz));
data_Pt=data_Pta.tolist()

########################################
#### Colocar na GUI pro usuário escolher uma de 4 opções:  Plotar a figura 6 completa ou apenas a força, ou apenas a velocidade ou apenas a potência ########################################
if plotar_fig_6==1:
	
	fig_6 = plt.figure()
	plt.subplot(3,1,1)
	#plt.plot(i1_values,v_R_x)
	for i1, color, label in zip(range(3), 'rgbk', 'XYZR'):
		plt.plot(i1_values, [i_F[i1] for i_F in data_Ft], color, label=label, lw=3)	
	#	plt.plot(i1_values, [F[i1] for F in data_Ft], color, label=label, lw=2)
	plt.grid(True)
	plt.xlabel('t(s)')
	plt.ylabel('Force (N)')
	plt.gca().set_ylim([data_Fta.min()*1.05,data_Fta.max()*1.05])
	plt.legend()
	#plt.show()
	
	plt.subplot(3,1,2)
	#plt.plot(i1_values,v_R_x)
	for i1, color, label in zip(range(3), 'rgbk', 'XYZR'):
		plt.plot(i1_values, [vt[i1] for vt in data_Vt], color, label=label, lw=3)
	plt.grid(True)
	plt.xlabel('t(s)')
	plt.ylabel('Velocity (m/s)')
	plt.gca().set_ylim([data_Vta.min()*1.05,data_Vta.max()*1.05])
	plt.legend()
	#plt.show()
	
	plt.subplot(3,1,3)
	#plt.plot(i1_values,v_R_x)
	for i1, color, label in zip(range(3), 'rgbk', 'XYZR'):
		plt.plot(i1_values, [P_i[i1] for P_i in data_Pt], color, label=label, lw=3)
	plt.grid(True)
	plt.xlabel('t(s)')
	plt.ylabel('Power (w)')
	plt.gca().set_ylim([data_Pta.min()*1.05,data_Pta.max()*1.05])
	plt.legend()
	plt.show()
	
####### figura 7, só a força ############
if plotar_fig_7==1:
	fig_7 = plt.figure()
	#plt.plot(i1_values,v_R_x)
	for i1, color, label in zip(range(3), 'rgbk', 'XYZR'):
		plt.plot(i1_values, [i_F[i1] for i_F in data_Ft], color, label=label, lw=3)	
	#	plt.plot(i1_values, [F[i1] for F in data_Ft], color, label=label, lw=2)
	plt.grid(True)
	plt.xlabel('t(s)')
	plt.ylabel('Force (N)')
	plt.gca().set_ylim([data_Fta.min()*1.05,data_Fta.max()*1.05])
	plt.legend()
	plt.show()
	
####### figura 8, só a velocidade #######
if plotar_fig_8==1:
	fig_8 = plt.figure()
	#plt.plot(i1_values,v_R_x)
	for i1, color, label in zip(range(3), 'rgbk', 'XYZR'):
		plt.plot(i1_values, [vt[i1] for vt in data_Vt], color, label=label, lw=3)
	plt.grid(True)
	plt.xlabel('t(s)')
	plt.ylabel('Velocity (m/s)')
	plt.gca().set_ylim([data_Vta.min()*1.05,data_Vta.max()*1.05])
	plt.legend()
	plt.show()
	
####### figura 9, só a potência #######
if plotar_fig_9==1:
	fig_9 = plt.figure()
	#plt.plot(i1_values,v_R_x)
	for i1, color, label in zip(range(3), 'rgbk', 'XYZR'):
		plt.plot(i1_values, [P_i[i1] for P_i in data_Pt], color, label=label, lw=3)
	plt.grid(True)
	plt.xlabel('t(s)')
	plt.ylabel('Power (w)')
	plt.gca().set_ylim([data_Pta.min()*1.05,data_Pta.max()*1.05])
	plt.legend()
	plt.show()

#########################################
######## Define sub rotina: onoff #######
#########################################
def onoff(t,Vy,thresh):
	maxthresh = []
	minthresh = []
	amostra_on = []
	amostra_off = []
	off1 = []
	t2=t[1:len(t)-1]
	n_on=len(t)-1

	for x in t2:
		if (Vy[x]>=thresh) & (Vy[x - 1] < thresh) & (Vy[x + 1] > thresh):
			amostra_on.append(x)
			n_on=x 
		
		elif (Vy[x]>=thresh) & (x>n_on) & (Vy[x - 1] > thresh) & (Vy[x + 1] < thresh):
			amostra_off.append(x)

	return amostra_on, amostra_off
#########################################
########### FIM rotina: onoff ###########
#########################################

#########################################
####### Define sub rotina: picodet ######
#########################################
def picodet(Vy,amostra_on,amostra_off):
	
	vetor_t_instant=[]
	vetor_y_instant=[]
	vetor_picos=[]
	n_vetor=0

	for n_vetor in (array(range(len(amostra_on)))):
		vetor_t_instant=list(range(amostra_on[n_vetor],amostra_off[n_vetor]+1))
		vetor_y_instant=Vy[vetor_t_instant]
		vetor_picos.append(max(vetor_y_instant))
	return vetor_picos
#########################################
########## FIM rotina: picodet ##########
#########################################

# Definir Sinal de Input
amostras = (array(range(len(data_Pz))))
thresh_Fz = 0.5*max(data_Fz)
thresh_Vz = 0.5*max(data_Vz)
thresh_Pz = 0.5*max(data_Pz)

amostra_on_Fz, amostra_off_Fz = onoff(amostras, data_Fz,thresh_Fz)
amostra_on_Vz, amostra_off_Vz = onoff(amostras, data_Vz,thresh_Vz)
amostra_on_Pz, amostra_off_Pz = onoff(amostras, data_Pz,thresh_Pz)


vetor_picos_Fz = picodet(data_Fz,amostra_on_Fz,amostra_off_Fz)
vetor_picos_Vz = picodet(data_Vz,amostra_on_Vz,amostra_off_Vz)
vetor_picos_Pz = picodet(data_Pz,amostra_on_Pz,amostra_off_Pz)

#print('vetor_picos_Fz = ',vetor_picos_Fz)
#print('vetor_picos_Vz = ',vetor_picos_Vz)
#print('vetor_picos_Pz = ',vetor_picos_Pz)

#########################################
########### Resultados Finais ###########
#########################################

mean_picos_Fz=mean(vetor_picos_Fz)
mean_picos_Vz=mean(vetor_picos_Vz)
mean_picos_Pz=mean(vetor_picos_Pz)

std_picos_Fz=std(vetor_picos_Fz)
std_picos_Vz=std(vetor_picos_Vz)
std_picos_Pz=std(vetor_picos_Pz)

max_picos_Fz = 0;
max_picos_Vz = 0;
max_picos_Pz = 0;

if vetor_picos_Fz:
	max_picos_Fz=max(vetor_picos_Fz)
if vetor_picos_Vz:
	max_picos_Vz=max(vetor_picos_Vz)
if vetor_picos_Pz:
	max_picos_Pz=max(vetor_picos_Pz)

######## Imprimir Resultados ########
print('#############################')
print('#############################')
print('___________RESULTS___________')
print('#############################')
print('__________FORCE_PEAKS________')
print('Mean = ',mean_picos_Fz)
print('Std = ',std_picos_Fz)
print('Max = ',max_picos_Fz)
print('#############################')
print('________VELOCITY_PEAKS_______')
print('Mean = ',mean_picos_Vz)
print('Std = ',std_picos_Vz)
print('Max = ',max_picos_Vz)
print('#############################')
print('__________POWER_PEAKS________')
print('Mean = ',mean_picos_Pz)
print('Std = ',std_picos_Pz)
print('Max = ',max_picos_Pz)
print('#############################')
print('#############################')
