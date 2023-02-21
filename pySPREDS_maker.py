'''
	This code lets you generate SPatially Resolved Dyamic Spectra (SPREDS) for N regions in the image. N can be any integer. Enter the regions in casa format in regions list.The code uses the fits TB images and make TB Dynamic spectra (DS).   

### Logic and inputs #####

1. User has to input the regions in the CASA image under 'regions' list. 
2. One has to provide frequency resolution, starttime, endtime and time resolution in the images.
3. The code computes the frequency and time arrays based on this for which the images have been made. 
#### IMPORTANT : The image names should follow the convention, Eg: 'Sun_$$$$_$$$$_061214.0_8~11_$$$$.fits' . Here 061214.0 is the starttime of the image data. 8~11 is the range of avg ing in the image across spw channels. #####

4. The code makes a dictionary to match the starttime and spw window to the array index (M,N) of the DS matrix.
   That means : It generates inially a time array based on your input (Eg: time_array=[061202.0,061202.5, 061203.0, ....]). Then a spw channel array is generated based on the user input on frequency averaging used (Eg: frq_array=['8~11', '12~15', ....]). Now the DS will have dimension of (len(time_array),len(frq_array)). Knowing the starttime and the spw range from each imagename one can now match it to the index of starttime and spw range in the arrays , time_array and frq_array. These indices is nothing but the (M,N) or the tuple of location where the net flux has to be recorded in the DS matrix.

5. The code calculates the peak TB in the regions specified by the user for every single image using tasks in image analysis toolkit of CASA, where in one can mask specific regions of the image and get the local statistics. The calculated net TB of a region would be recorded at the (M,N) index of the DS matrix. 

6. The SPREDS matrix is made iteratively for all the regions specified by user   

######## NOTE: SPREDS will be made till the last time stamp excluding it. This is because usually no data exists for the last few seconds anyway and if one wants to do continuous SPREDS for consecutive times, then merging one with the adjacent observation period is easier if last time stamp is omitted. Eg 06:12:02.0 - 06:16:02, 06:16:02 - 06:20:02.0. Its better the former SPREDS didn't have a column of 0s for 06:16:02.0. 

'''

import numpy as np
import matplotlib.pyplot as plt
import glob,os
import datetime as dt
import pickle
#from copy import deepcopy
import numpy.ma as ma
from numpy import matrix
from astropy.io import fits
from multiprocessing import Pool
############### INPUT ####################
#regions=[(176.,216.),(187.,211.),(215.,197.),(226.,191.),(203.,204.),(192.,194.)] #Give centre pixel coordinates of the regions. A2=(189,195), Burst=(193,199) ; A1=(202,206) in H band trueloc img cutouts. regions=[(203.,206.),(193.,199.)]
regions=[]# LEave blank if region has to be dynamically allocated
reg_name=['A1']
reg_nsig=[2] # The size of the region in units of beam major axis size. This will the ellipse size used to find SPREDS at a chosen x0,y0
'''
The fluxes are recorded using a masking technique where images are multiplied by an elliptical mask
Make sure image size of all images are same before choosing common masks. If image sizes are different give specific_img_mask==True which mends the mask size as per image size before.
'''
dominant_source=True # Don't give regions list if this is true. Code will dynamically find the location of the peak source.
use_mean_psfsize=False # If true code finds a mean size for psf across frequency band and use that to get region size
specific_img_mask=True # If you need psf to be found for every image and mask to be made accordingly. Else the code will either use a frequency dependent mask by making a mask for each frequency using a representative image or a global mask for all images in the folder if use_mean_psfsize==True
source_reg=[] # Provide the search box extent where to find dominent source [(blc_x,blc_y),(trc_x,trc_y)]. Leave it blank list if dominant source is not to be found across the full image or if dominant_source=False
savemasked_SPREDS= [True] # Give an array of flags each corresponding to a region. If True, the final SPREDS image, saved as eps, would mask all TB measurements less than 5 sigma of the image noise.
#regions=['circle[[1622.66pix,1612.23pix],4.5pix]']
reftimes=['061602.0','061802.0','061950.0']# LEave [] for the code to chose start time mid and end time as ref times for finding mean frequency dependent psf size. If you give timesstamps when to find the size of psf, make sure all frequencies have some image at this time stamp.
TBfitsdir='/Data1/atul/20141103/Solar_Event/1099030576-150-161/All_fits_K_05_160kHz_I/'
is_uncalib_imgs=False # If true the code will work for uncalibrated maps also. Else Use TB maps alone
starttime=dt.datetime(2014,11,3,6,16,2) # start time
endtime=dt.datetime(2014,11,3,6,20,2)
tdt=dt.timedelta(seconds=0.5)# Time resolution of images in seconds
delt=0.5
ncoarse=12  # ncoarse_chan=2 for picket fence, 12 for harmonic and 24 for continuous mode observations
freq_res=40 # freq resolution of the data in kHz
strtfrq=191.36 # 110.72 an 222.08 , F and H
chnwid=0.04 # Channel width in MS visibility data. i.e. gao between two adjacent CASA spws in MHz.
avg_chn=4
ynbins=8 # No. of bins needed in the axis. This is 1 minus the number of yticks. Set this as a multiple of the length of the y axis of your Dynamic spectrum to be made. Eg: 48 spw ranges are there in my localised DS , then code will set ynbins as 8 or 6 which divides 48.
xnbins=4 
DS_folder='/Data1/atul/20141103/Solar_Event/1099030576-150-161/Analysis/Jy_SPREDS/' # Folder to save the Dyn spec pickle file.Code will make this
nsig=4 # What sigma limit do you wanna put on the minimum believable fluxes in the DS. thresh_DSfile will record nsig*std_dev of noise for each M,N entry of the Dynamic spectrum. One can use this later to flag unwanted entries in the DS.
scaling=10**-4 # Value by which the DS has to be multiplied to make visualisation better. ##!! Note that data is never scaled to whatever you specify here. This scaling is used just while plotting data.
collab_lab=r'$SFU$'
want_in_Jy=True # True if DS written out has to be in Jy units 
peakFlx_SPREDS=False # If you want to record peak Tb or Jy/beam rather than mean Tb or mean Jy for the chosen region.
Error_boxes=[] # Leave this blank list [] if you prefer the code to autoset the error box dimension. The autoset would do it as [(50,50),(imsize-50,imsize/4)]. The list has 2 elements. [(blcx,blcy),(trcx,trcy)]. i.e bottom left corner and top right corner coordinates.
cmap=None # Chose a colormap or leave as None for default matplotlib colormap
plot_time_inUTC=True #Plot time in UTC or seconds
xax_rotation=45 # plot axis label rotation
### Input for Confidance interval method ###################

want_errCI=True # If this is true only a single error Inerval will be made using data in all the Error_boxes provided. the interval will be computed by looking at the statistics of the error points within all boxes. The alpha % confidence interval for error values will be found and 2 errDS will be made 1 for lower limit and 1 for upper limit of confidance interval.
alpha=0.05 # in decimals. if 0.05 is chosen , it means find the 95% confidance interval for the error distribution.
snr=4 # SNR cut for data to be declared useful and unmasked
###################################################################

nproc=max(len(reg_name),10) # Number of processors utilised in parallel
imgtitle=['Dynamic Spectrum (Region: '+i+')' for i in reg_name]
imgfilname=['Reg'+i+'_DS' for i in reg_name] 
#DS_folder='DS_test/'
DSpfile=[i+'.p' for i in imgfilname] # Name of the Dynamic spectral pickle file.
thresh_DSfile=['min_Jy_'+i+'_DS.p' for i in reg_name]
os.system('mkdir '+DS_folder)
################## TIME array making #########################
os.chdir(TBfitsdir)
print imgtitle
# Generating a dictionary to match the timestamp and the DS index 
wholerng=[]
stt=starttime
DSx=[]
i=0
time_xax=[]

while stt<endtime:
	tempt=stt.time().strftime('%H:%M:%S.%f')[:-5]
	stt+=tdt
	time_xax+=[tempt]
	wholerng+=[tempt.replace(':','')]
	DSx+=[i]
	i+=1
#wholerng=[val.replace('.','') for val in wholerng]
time_DSx=dict(zip(wholerng,DSx))
time_xax+=[stt.time().strftime('%H:%M:%S.%f')[:-5]]
######################## Frequency array generation ##################
fqs=[]
yax=[] # This is for plotting purpose. We need the frequencies corresponding to each row of the Dynamic spectrum.
minchan=int(1.28*10**3/freq_res) # 1.28MHz is a course channel width of MWA
M = (minchan/4)-1    # This is the last flagged channel at the start.Also the half-1 of the net allowed channel width.
a = (M+1)*2       # This is the middle of a coarse channel which is bad. 
flagwidth=a    # Width of the channels that are flagged in total at the edge of 2 adjoint channels. This is = minchan
DEF_STR='0:'
strt_chan=M+1        # 1st useful/unflagged channel
#print minchan,a,M,flagwidth
Total_channels=1.28*10**3*ncoarse/freq_res



while strt_chan<Total_channels:
	spw=DEF_STR+str(strt_chan)+'~'+str(a-1)+';'
	spw=spw+str(a+1)+'~'+str(strt_chan+flagwidth-1)+';'
	strt_chan=strt_chan+minchan  # Moving the start of good channels to the next block of coarse channel
	a+=minchan		# moving a to the middle of the next coarse channel
	DEF_STR=spw
spw=spw[:-1]
fqrs=spw.split(':')[1].split(';')


#Creating spw windows for channel averaging
indx=[]
count=0
for i in np.arange(len(fqrs)): 
	tmp=fqrs[i]
	ok=1
	ptr=int(tmp.split('~')[0])
	endptr=int(tmp.split('~')[1])
	while ok==1: 
		stf=ptr
		nxt=min(ptr+avg_chn-1,endptr)
		fqs+=[str(stf)+'~'+str(nxt)]
		yax+=[(stf+nxt)/2.]
		indx+=[count]
		count+=1
		ptr+=avg_chn
		if ptr>=endptr:
			ok=0
fqmatch=dict(zip(fqs,indx))
print 'Frequency ranges where images are to be made: ', fqs
############################ Mean PSF size ####################################################
#Calculating a mean size of the PSF major and minor axis using near midtime spectroscopic images
a_mean=0
b_mean=0
posang_mean=0
imgls=[]
if use_mean_psfsize==True:
	inx=0
	while len(imgls)<len(fqs)/2: # Checking if atleast half the fqs ranges are covered
		imgls=glob.glob('*'+wholerng[int(len(wholerng)/2.)+inx]+'*.fits')
		inx+=1
	for im in imgls:
		head=fits.getheader(im)
		cellsize=abs(head['CDELT2'])*3600
		bmaj=head['BMAJ']*3600/cellsize # in degrees --> pix Full maj axis extent
		bmin=head['BMIN']*3600/cellsize # ''
		a_mean+=bmaj/2.
		b_mean+=bmin/2.
		posang_mean+=head['BPA']*np.pi/180
	a_mean=a_mean/len(imgls)
	b_mean=b_mean/len(imgls)
	posang_mean=posang_mean/len(imgls)
	Imsize=head['NAXIS1']
	print 'Global image size: ',Imsize
############################Computing the DS####################################################
def SPREDS(i):
	global Error_boxes
	DS=np.zeros((len(fqs),len(wholerng)))
	if want_errCI==False:
		errDS=np.zeros(DS.shape)
	elif i==0:
		errDS_up=np.zeros(DS.shape)
		errDS_lo=np.zeros(DS.shape)

	print '\n###############################\nRegion of choice :',reg_name[i] 
	print 'Size of the Dynamic spectrum: ',len(fqs),len(wholerng)
	print 'Total no. of images: ',len(imgs)
	countdown=len(imgs)
	########################## Ellipse mask maker ########################
	rsig=reg_nsig[i]
	# Make mask centred at the given region centre in variable named 'regions'. If not specified and dominant_source==True then a nominal mask is made centred at the bright point in the first image. Mask centre is Mx0,My0 	
	if use_mean_psfsize==True:
		if dominant_source==True:
			tdat=fits.getdata(imgs[0])[0][0]# A representative img for making mask
			if len(source_reg)==0: # Find the location of the maximum flux in one img
				maxloc_pix=np.where(tdat==tdat.max())
				Mx0=maxloc_pix[0][0] # Mask centre
				My0=maxloc_pix[1][0]
			else:
				(blcx1,blcy1),(trcx1,trcy1)=source_reg
				maxpy,maxpx=np.where(tdat[blcy1:trcy1,blcx1:trcx1]==np.max(tdat[blcy1:trcy1,blcx1:trcx1]))
				Mx0=blcy1+maxpy[0]
				My0=blcx1+maxpx[0]
				if Mx0 in [blcy1,trcy1-1] or My0 in [blcx1,trcx1-1]: # Checking if the maxima point is some source outside the seource_reg loading flux to this region. In such a case the maximum point will the at the edge of the box. So checking if Mx0,My0 is in the box boundary. trcx-1 because the the slicing rules in python doesnt include the trcx^th cell.
					Mx0=int((blcy1+trcy1)/2.)
					My0=int((blcx1+trcx1)/2.)
		else:
			cent=regions[i]
			Mx0=cent[1]
			My0=cent[0]			

		R=np.array([[np.cos(posang_mean),-np.sin(posang_mean)],[np.sin(posang_mean),np.cos(posang_mean)]])
		a_mean=bmaj/2.*rsig # Chosing nsig region for calculating flux density
		b_mean=bmin/2.*rsig
		beam_ar=np.pi*a_mean*b_mean/np.log(2)/rsig**2
		# Making mask image #################################
		mask_img=np.zeros((Imsize,Imsize))
		m=mask_img.shape[0]
		n=mask_img.shape[1]
		print 'Shape of mask: ',mask_img.shape
		R=np.matrix(R)
		for k in range(m):
			for j in range(n):
				x_v=np.matrix([[k-Mx0],[j-My0]])
				xp_v=R*x_v
				xp_v=np.array(xp_v.transpose()).reshape(2)
				if xp_v[0]**2/a_mean**2 + xp_v[1]**2/b_mean**2<1:
					mask_img[k,j]=1
		print 'Fixed Global Mask made..',' Centre: (',My0,',',Mx0,')'

	dyn_errbox=0
	if Error_boxes==[]:
		dyn_errbox=1

	if want_errCI==False and Error_boxes!=[]:
		Error_box=Error_boxes[i]
		blcx=Error_box[0][0]
		blcy=Error_box[0][1]
		trcx=Error_box[1][0]
		trcy=Error_box[1][1]

	for fq in fqs:
		print '\n Channel: ',fq,'\n'
		imgs1=sorted([ig for ig in imgs if '_'+fq+'_' in ig])
		print 'No. of images in ',fq,' ',len(imgs1)
		if len(imgs1)==0:
			print 'Bad channel.. No images.!!!\n'
			continue
		## Generating a image mask for this spw range		
		if specific_img_mask==False and use_mean_psfsize==False:
			if fq!=fqs[0]: # Delete the old mask matrix
				del mask_img
			if ref_times==[]:
				timps=[wholerng[0],wholerng[int(len(wholerng)/2.)],wholerng[-10]] # 3 equally separated timesteps. Images for this fq during this time ill be used to get a mean size of PSF.
			else:
				timps=ref_times
			a_mean=0
			b_mean=0
			posang_mean=0
			imgls=[ig for ig in imgs1 if ig.split('_')[3] in timps]
			for imt in imgls:
				head=fits.getheader(imt)
				cellsize=abs(head['CDELT2'])*3600
				bmaj=head['BMAJ']*3600/cellsize # in degrees --> pix Full maj axis extent
				bmin=head['BMIN']*3600/cellsize # ''
				a_mean+=bmaj/2.
				b_mean+=bmin/2.
				posang_mean+=head['BPA']*np.pi/180
			a_mean=a_mean/len(imgls)
			b_mean=b_mean/len(imgls)
			posang_mean=posang_mean/len(imgls)
			Imsize=head['NAXIS1']	
			if dominant_source==True:
				tdat=fits.getdata(imgs1[0])[0][0]# A representative img for making mask
				if len(source_reg)==0: # Find the location of the maximum flux in one img
					maxloc_pix=np.where(tdat==tdat.max())
					Mx0=maxloc_pix[0][0] # Mask centre
					My0=maxloc_pix[1][0]
				else:
					(blcx1,blcy1),(trcx1,trcy1)=source_reg
					maxpy,maxpx=np.where(tdat[blcy1:trcy1,blcx1:trcx1]==np.max(tdat[blcy1:trcy1,blcx1:trcx1]))
					Mx0=blcy1+maxpy[0]
					My0=blcx1+maxpx[0]
					if Mx0 in [blcy1,trcy1-1] or My0 in [blcx1,trcx1-1]: # Checking if the maxima point is some source outside the seource_reg loading flux to this region. In such a case the maximum point will the at the edge of the box. So checking if Mx0,My0 is in the box boundary. trcx-1 because the the slicing rules in python doesnt include the trcx^th cell.
						Mx0=int((blcy1+trcy1)/2.)
						My0=int((blcx1+trcx1)/2.)

			else:
				cent=regions[i]
				Mx0=cent[1]
				My0=cent[0]			
			R=np.array([[np.cos(posang_mean),-np.sin(posang_mean)],[np.sin(posang_mean),np.cos(posang_mean)]])
			a_mean=bmaj/2.*rsig # Chosing nsig region for calculating flux density
			b_mean=bmin/2.*rsig
			beam_ar=np.pi*a_mean*b_mean/np.log(2)/rsig**2
			# Making mask image #################################
			mask_img=np.zeros((Imsize,Imsize))
			m=mask_img.shape[0]
			n=mask_img.shape[1]
			print 'Shape of mask: ',mask_img.shape
			R=np.matrix(R)
			for k in range(m):
				for j in range(n):
					x_v=np.matrix([[k-Mx0],[j-My0]])
					xp_v=R*x_v
					xp_v=np.array(xp_v.transpose()).reshape(2)
					if xp_v[0]**2/a_mean**2 + xp_v[1]**2/b_mean**2<1:
						mask_img[k,j]=1
			print 'Mask made for spw: ',fq,' Centre: (',My0,',',Mx0,')'

		for img in imgs1:
			print 'Analysing image: ',img
			ts=img.split('_')[3]
			spw=img.split('_')[4]
			M=fqmatch[spw]
			N=time_DSx[ts]
			data=fits.getdata(img)[0][0]
			head=fits.getheader(img)
			imsize=data.shape[0]
			try:
				cellsize=float(head['CELL'])
			except:
				cellsize=abs(head['CDELT2'])*3600
			################## Error box definition  ########################
			if dyn_errbox==1:
				Error_boxes=[[(50,50),(imsize-50,imsize/4)]]
				blcx=Error_boxes[0][0][0]
				blcy=Error_boxes[0][0][1]
				trcx=Error_boxes[0][1][0]
				trcy=Error_boxes[0][1][1]

			####################################################################
			if len(regions)!=0:
				X0=regions[i][1]
				Y0=regions[i][0]		

			if dominant_source==True or specific_img_mak==True:
				if len(source_reg)==0:
					maxloc_pix=np.where(data==data.max())
					X0=maxloc_pix[0][0]
					Y0=maxloc_pix[1][0]
				else:
					(blcx1,blcy1),(trcx1,trcy1)=source_reg
					maxpy,maxpx=np.where(data[blcy1:trcy1,blcx1:trcx1]==np.max(data[blcy1:trcy1,blcx1:trcx1]))
					X0=blcy1+maxpy[0]
					Y0=blcx1+maxpx[0]

					if X0 in [blcy1,trcy1-1] or Y0 in [blcx1,trcx1-1]: # Checking if the maxima point is some source outside the seource_reg loading flux to this region. In such a case the maximum point will the at the edge of the box. So checking if Mx0,My0 is in the box boundary. trcx-1 because the the slicing rules in python doesnt include the trcx^th cell.
						X0=int((blcy1+trcy1)/2.)
						Y0=int((blcx1+trcx1)/2.)
				if specific_img_mask==True:
					posang=head['BPA']*np.pi/180 # in degrees --> radian
					bmaj=head['BMAJ']*3600/cellsize # in degrees --> pix Full maj axis extent
					bmin=head['BMIN']*3600/cellsize # ''
					R=np.array([[np.cos(posang),-np.sin(posang)],[np.sin(posang),np.cos(posang)]])
					a=bmaj/2.*rsig # Chosing nsig region for calculating flux density
					b=bmin/2.*rsig
					beam_ar=np.pi*a*b/np.log(2)/rsig**2
					# Making mask image #################################
					Mask_Img=np.zeros((imsize,imsize))
					m=Mask_Img.shape[0]
					n=Mask_Img.shape[1]
					print 'Shape of mask: ',Mask_Img.shape
					R=np.matrix(R)
					for k in range(m):
						for j in range(n):
							x_v=np.matrix([[k-X0],[j-Y0]])
							xp_v=R*x_v
							xp_v=np.array(xp_v.transpose()).reshape(2)
							if xp_v[0]**2/a**2 + xp_v[1]**2/b**2<1:
								Mask_Img[k,j]=1
					print 'Specific Mask made..',ts,' ',spw
				else: # We need to shift our mask appropriately to the new centre.
					dX=X0-Mx0
					dY=Y0-My0
					top,bottom,left,right=0,0,0,0
					if dY>0:
						left=dY
						y0=0
					else:
						right=-1*dY
						y0=right
					if dX>0:
						top=dX
						x0=0
					else:
						bottom=-1*dX
						x0=bottom
					mask_imgt=np.pad(mask_img,((top,bottom),(left,right)),'constant')
					Mask_Img=mask_imgt[x0:x0+imsize,y0:y0+imsize]
					print 'Mask shifted to new maxima at,',' (',Y0,',',X0,')'
					print 'Mask shape: ',Mask_Img.shape
					del mask_imgt
				msk_data=data*Mask_Img # Make masked data matrix
				del Mask_Img
			else:
				msk_data=data*mask_img
			##########################################################
			#flx=np.sum(msk_data)/beam_ar	
			noise=np.array([])
			if want_in_Jy==False:
				if peakFlx_SPREDS==False:
					msk_data=ma.masked_equal(msk_data,0)
					flx=np.mean(msk_data)
				else:
					flx=np.max(msk_data)
				if want_errCI==False:
					rms_noise=np.std(data[blcy:trcy,blcx:trcx])
				elif i==0:# If Confidence interval method is opted , we need only 1 error estimation for all error boxes. So only for 1st region we need to calculate error DS
					for Error_box in Error_boxes:
						blcx=Error_box[0][0]
						blcy=Error_box[0][1]
						trcx=Error_box[1][0]
						trcy=Error_box[1][1]			
						noise=np.append(noise,data[blcy:trcy,blcx:trcx].flatten())
					noise=np.sort(noise)
					lowlim,uplim=noise[int((alpha/2.0)*len(noise))],noise[int((1-alpha/2.0)*len(noise))]

			else:
				if is_uncalib_imgs==False:
					factK=head['Conversion factor']
					fact=head['Attenuator Correction factor']
				else:
					factK,fact=1,1
				if peakFlx_SPREDS==False:
					flx=np.sum(msk_data)/beam_ar*fact/factK
				else:
					flx=np.max(msk_data)*fact/factK # Flux in Jy/beam
				if want_errCI==False:
					rms_noise=np.std(data[blcy:trcy,blcx:trcx])*fact/factK/np.sqrt(reg_nsig[i])	
				elif i==0:		
					for Error_box in Error_boxes:
						blcx=Error_box[0][0]
						blcy=Error_box[0][1]
						trcx=Error_box[1][0]
						trcy=Error_box[1][1]					
						noise=np.append(noise,(data[blcy:trcy,blcx:trcx]*fact/factK).flatten())	
					noise=np.sort(noise)
					lowlim,uplim=noise[int((alpha/2.0)*len(noise))],noise[int((1-alpha/2.0)*len(noise))]

			print 'Value recorded in the DS at (',M,',',N,'): ',flx*scaling,' ',collab_lab.replace('$',''),'for ',ts,' ',spw
			DS[M,N]=flx
			#st1=ia.statistics(region=errbox)
			if want_errCI==False:
				errDS[M,N]=nsig*rms_noise
				print nsig,' sigma noise value stored in error DS at (',M,',',N,') : ',errDS[M,N]*scaling,' ',collab_lab.replace('$',''),'for ',ts,' ',spw
			elif i==0:
				errDS_up[M,N]=uplim
				errDS_lo[M,N]=lowlim
				print 'Conf Interval for noise value stored in error DS at (',M,',',N,') : ',errDS_lo[M,N]*scaling,' - ',errDS_up[M,N]*scaling,' ',collab_lab,'for ',ts,' ',spw
			countdown-=1
			print '############## Images left : ',countdown,' #############'
			del data
			del msk_data
	pickle.dump(DS,open(DS_folder+DSpfile[i],'wb'))
	print 'SPREDS for Region ',i+1,', ',reg_name[i],' written out!!'
	if want_errCI==False:
		pickle.dump(errDS,open(DS_folder+thresh_DSfile[i],'wb'))	
		print 'Error DS written out for region ',reg_name[i],'!!'		
		del errDS
	elif i==0:
		pickle.dump(errDS_up,open(DS_folder+'ErrDS_CI_uplim.p','wb'))			
		pickle.dump(errDS_lo,open(DS_folder+'ErrDS_CI_lowlim.p','wb'))
		print 'Error DS written out using confidence interval method!!'			
		del errDS_up
		del errDS_lo

	del DS
########################## Plotting Section ##############################
if __name__=='__main__':
	#regions=regions[0:-1]
	Nregs=len(reg_name)
	hme=os.getcwd()
	imgs=glob.glob('*.fits')
	p=Pool(nproc)
	os.nice(20)
	succ=p.map(SPREDS,np.arange(Nregs))
	yax1=strtfrq+np.array(yax)*chnwid
	ptr=0
	loc=0
	step=len(fqs)/ynbins
	yp=[]   # The pointers denoting where the labels are to be planted. 
	yaxp=[] # Axis labels for the y axis.

	ptrx=0
	locx=0
	stepx=len(time_xax)/xnbins
	xp=[]
	xaxp=[]
	if plot_time_inUTC==True:
		xax1=time_xax
	else:
		xax1=np.arange(len(time_xax))*0.5

	while ptrx<=len(xax1):
		xaxp+=[xax1[locx]]
		xp+=[locx]
		ptrx+=stepx
		locx=ptrx

	while ptr<=len(yax1):
		yaxp+=[yax1[loc]]
		yp+=[loc]
		ptr+=step
		loc=ptr-1
	yaxp=[int(round(val,0)) for val in yaxp]
	if plot_time_inUTC==False:
		xaxp=[int(round(val,0)) for val in xaxp]
	i=0
	# Setting values<nsig noise threshold to 0. These will later be masked.
	if want_errCI==True:
		errDS_up=pickle.load(open(DS_folder+'ErrDS_CI_uplim.p','rb'))
		errDS_lo=pickle.load(open(DS_folder+'ErrDS_CI_lowlim.p','rb'))
		plt.figure()
		plt.imshow(errDS_up*scaling,origin='lower',aspect='auto',cmap=cmap)
		plt.yticks(yp,yaxp)
		plt.xticks(xp,xaxp,rotation=xax_rotation)
		plt.ylabel('Frequency (MHz)',size=16)
		plt.xlabel('Time (s)',size=16)
		plt.xticks(size=14)
		plt.yticks(size=14)
		h=plt.colorbar()
		h.set_label(collab_lab,size=17)
		h.ax.tick_params(labelsize=12)
		plt.savefig(DS_folder+'ErrDS_uplim.png',bbox_inches='tight')
		plt.savefig(DS_folder+'ErrDS_uplim.eps',format='eps',bbox_inches='tight')
		plt.title(imgtitle[i],size=18)
		plt.savefig(DS_folder+'ErrDS_uplim_titled.png',bbox_inches='tight')
		plt.savefig(DS_folder+'ErrDS_uplim_titled.eps',format='eps',bbox_inches='tight')
		plt.close()
		plt.figure()
		plt.imshow(errDS_lo*scaling,origin='lower',aspect='auto',cmap=cmap)
		plt.yticks(yp,yaxp)
		plt.xticks(xp,xaxp,rotation=xax_rotation)
		plt.ylabel('Frequency (MHz)',size=16)
		plt.xlabel('Time (s)',size=16)
		plt.xticks(size=14)
		plt.yticks(size=14)
		h=plt.colorbar()
		h.set_label(collab_lab,size=17)
		h.ax.tick_params(labelsize=12)
		plt.savefig(DS_folder+'ErrDS_lolim.png',bbox_inches='tight')
		plt.savefig(DS_folder+'ErrDS_lolim.eps',format='eps',bbox_inches='tight')
		plt.title(imgtitle[i],size=18)
		plt.savefig(DS_folder+'ErrDS_lolim_titled.png',bbox_inches='tight')
		plt.savefig(DS_folder+'ErrDS_lolim_titled.eps',format='eps',bbox_inches='tight')
		plt.close()
	
		errDS=np.abs(errDS_up-errDS_lo)*snr
	for i in range(Nregs):
		DS=pickle.load(open(DS_folder+DSpfile[i],'rb'))
		if want_errCI==False:
			errDS=pickle.load(open(DS_folder+thresh_DSfile[i],'rb'))
			plt.figure()
			plt.imshow(errDS*scaling,origin='lower',aspect='auto',cmap=cmap)
			plt.yticks(yp,yaxp)
			plt.xticks(xp,xaxp,rotation=xax_rotation)
			plt.ylabel('Frequency (MHz)',size=16)
			plt.xlabel('Time (s)',size=16)
			plt.xticks(size=14)
			plt.yticks(size=14)
			h=plt.colorbar()
			h.set_label(collab_lab,size=17)
			h.ax.tick_params(labelsize=12)
			plt.savefig(DS_folder+'ErrDS.png',bbox_inches='tight')
			plt.savefig(DS_folder+'ErrDS.eps',format='eps',bbox_inches='tight')
			plt.title(imgtitle[i],size=18)
			plt.savefig(DS_folder+'ErrDS_titled.png',bbox_inches='tight')
			plt.savefig(DS_folder+'ErrDS_titled.eps',format='eps',bbox_inches='tight')
			plt.close()

		if savemasked_SPREDS[i]==True:
			for k1 in range(len(fqs)):
				for k2 in range(len(wholerng)-1):
					if DS[k1,k2]<errDS[k1,k2]:
						DS[k1,k2]=0
			DSm=ma.masked_equal(DS,0)
		else:
			DSm=DS
		plt.figure()
		plt.imshow(DSm*scaling,origin='lower',aspect='auto',cmap=cmap)
		plt.yticks(yp,yaxp)
		plt.xticks(xp,xaxp,rotation=xax_rotation)
		plt.ylabel('Frequency (MHz)',size=16)
		plt.xlabel('Time (s)',size=16)
		plt.xticks(size=14)
		plt.yticks(size=14)
		h=plt.colorbar()
		h.set_label(collab_lab,size=17)
		h.ax.tick_params(labelsize=12)
		plt.savefig(DS_folder+imgfilname[i]+'.png',bbox_inches='tight')
		plt.savefig(DS_folder+imgfilname[i]+'.eps',format='eps',bbox_inches='tight')
		plt.title(imgtitle[i],size=18)
		plt.savefig(DS_folder+imgfilname[i]+'_titled.png',bbox_inches='tight')
		plt.savefig(DS_folder+imgfilname[i]+'_titled.eps',format='eps',bbox_inches='tight')
		plt.close()
	
	os.chdir(hme)
