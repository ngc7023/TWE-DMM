import codecs
import numpy as np
import matplotlib.pyplot as plt

def getphi(phifile):
	f=codecs.open(phifile,'r')
	phi=[]
	for line in f:
		tokens=line.strip().split()
		tmp=[float(t) for t in tokens]
		tmp.sort()
		phi.append(tmp[::-1])
	f.close()
	phi=np.array(phi)
	return phi

def getMAM(phi):
	phiT=phi.T
	phiT=np.delete(phiT, [2,6], axis = 1)
	maxp=[]
	minp=[]
	avgp=[]
	x=[]
	for i in range(0,len(phiT),10):
		tmp=sum(phiT[0:i+1,:])
		maxp.append(max(tmp))
		minp.append(min(tmp))
		avgp.append(sum(tmp)/len(tmp))
		x.append(i+1)
		if len(x)==500:
			break
	return maxp,minp,avgp,x

def plot(maxp,minp,avgp,x):
	fig, ax = plt.subplots()
	plt.subplot(111)
	label=['Maximum','Minimum','Average']
	plt.plot(x,maxp,'-.',label=label[0])
	plt.plot(x,minp,':',label=label[1])
	plt.plot(x,avgp,'-',label=label[2])
	plt.xlabel('number of words')
	plt.ylabel('sum of  probability')
	fig.legend(loc='upper center',bbox_to_anchor=(0.80, 0.30),ncol=1,columnspacing=0.1)
	plt.show()

def plt_all(phi):
	fig, ax = plt.subplots()
	plt.subplot(111)
	sum_value=[]
	for i in range(0, len(phi)):
		# if i==2 or i==6:
		# 	continue
		tmp=[]
		count=10
		while count<=500:
			tmp.append(sum(phi[i][:count]))
			count+=10
		x=list(range(len(tmp)))
		plt.plot(x,tmp,label=str(i))
	fig.legend(loc = 'upper center', bbox_to_anchor = (0.5, 0.89), ncol = 3, columnspacing = 0.1)
	plt.show()


if __name__ == '__main__':
	phifile='../RE/TWE_DMM/k_10/phifile.txt'
	phi=getphi(phifile)
	maxp,minp,avgp,x=getMAM(phi)
	plot(maxp,minp,avgp,x)
	# plt_all(phi)


