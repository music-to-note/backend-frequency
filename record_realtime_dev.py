import pyaudio
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("hacktech2019-233403-firebase-adminsdk-4frsw-f31a97f325.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://hacktech2019-233403.firebaseio.com/'
})

ref = db.reference('/')




plt.style.use('ggplot')

form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 44100#44100 # 22050 # 44.1kHz sampling rate
chunk = 16384
# chunk = 44100 # samples for buffer (more samples = better freq resolution)
# dev_index = 2 # device index found by p.get_device_info_by_index(ii)

audio = pyaudio.PyAudio() # create pyaudio instantiation

# mic sensitivity correction and bit conversion
mic_sens_dBV = -47.0 # mic sensitivity in dBV + any gain
mic_sens_corr = np.power(10.0,mic_sens_dBV/20.0) # calculate mic sensitivity conversion factor

# compute FFT parameters
f_vec = samp_rate*np.arange(chunk/2)/chunk # frequency vector based on window size and sample rate
mic_low_freq = 70 # low frequency response of the mic (mine in this case is 100 Hz)
low_freq_loc = np.argmin(np.abs(f_vec-mic_low_freq))

# prepare plot for live updating
plt.ion()
fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)
annot = ax.text(np.exp(np.log((0.8*f_vec[-1]))/2),0,"Measuring Noise...",\
                fontsize=30,horizontalalignment='center')
y = np.zeros((int(np.floor(chunk/2)),1))
line1, = ax.plot(f_vec,y)
plt.xlabel('Frequency [Hz]',fontsize=22)
plt.ylabel('Amplitude [Pa]',fontsize=22)
plt.grid(True)
plt.annotate(r'$\Delta f_{max}$: %2.1f Hz' % (samp_rate/(2*chunk)),xy=(0.7,0.9),xycoords='figure fraction',fontsize=16)
ax.set_xscale('log')
ax.set_xlim([1,0.8*samp_rate])
plt.pause(0.0001)

# create pyaudio stream
stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                    input = True, \
                    frames_per_buffer=chunk)

# some peak-finding and noise preallocations
peak_shift = 5
noise_fft_vec,noise_amp_vec = [],[]
annot_array,annot_locs = [],[]
annot_array.append(annot)
peak_data = []
noise_len = 5
ii = 0

# loop through stream and look for dominant peaks while also subtracting noise
while True:

    # read stream and convert data from bits to Pascals
    stream.start_stream()
    try:
        data = np.fromstring(stream.read(chunk),dtype=np.int16)
        if ii==noise_len:
            data = data-noise_amp
        data = ((data/np.power(2.0,15))*5.25)*(mic_sens_corr)
    except IOError as ex:
        # if ex[1] != pyaudio.paInputOverflowed:
        #     raise
        data = '\x00' * chunk
    stream.stop_stream()

    # compute FFT
    fft_data = (np.abs(np.fft.fft(data))[0:int(np.floor(chunk/2))])/chunk
    fft_data[1:] = 2*fft_data[1:]

    # calculate and subtract average spectral noise
    if ii<noise_len:
        if ii==0:
            print("Stay Quiet, Measuring Noise...")
        noise_fft_vec.append(fft_data)
        noise_amp_vec.extend(data)
        print(".")
        if ii==noise_len-1:
            noise_fft = np.max(noise_fft_vec,axis=0)
            noise_amp = np.mean(noise_amp_vec)
            print("Now Recording")
        ii+=1
        continue

    fft_data = np.subtract(fft_data,noise_fft) # subtract average spectral noise

    # plot the new data and adjust y-axis (if needed)
    line1.set_ydata(fft_data)
    if np.max(fft_data)>(ax.get_ylim())[1] or np.max(fft_data)<0.5*((ax.get_ylim())[1]):
        ax.set_ylim([0,1.2*np.max(fft_data)])

    # remove old peak annotations
    try:
        for annots in annot_array:
            annots.remove()
    except:
        pass
    # annotate peak frequencies (6 largest peaks, max width of 10 Hz [can be controlled by peak_shift above])
    annot_array = []
    peak_data = 1.0*fft_data


    peak_data = peak_data[:150]
    print(f_vec[149+low_freq_loc])
    for i in range(11):
        peak_data[low_freq_loc + i] += 0.0002 # 0.0005 - 0.001


    # print(peak_data[low_freq_loc:])


    freqs = []
    for jj in range(6):
        max_loc = np.argmax(peak_data[low_freq_loc:])
        # print(f_vec[max_loc+low_freq_loc])

        freqs.append(f_vec[max_loc+low_freq_loc])

        if peak_data[max_loc+low_freq_loc]>10*np.mean(noise_amp):
            annot = ax.annotate('Freq: %2.2f'%(f_vec[max_loc+low_freq_loc]),xy=(f_vec[max_loc+low_freq_loc],fft_data[max_loc+low_freq_loc]),\
                            xycoords='data',xytext=(-30,30),textcoords='offset points',\
                            arrowprops=dict(arrowstyle="->",color='k'),ha='center',va='bottom')
            if jj==3:
                annot.set_position((40,60))
            if jj==4:
                annot.set_x(40)
            if jj==5:
                annot.set_position((-30,15))

            annot_locs.append(annot.get_position())
            annot_array.append(annot)
            # zero-out old peaks so we dont find them again
            # peak_data[max_loc+low_freq_loc-peak_shift:max_loc+low_freq_loc+peak_shift] = np.repeat(0,len(peak_data[max_loc+low_freq_loc-peak_shift:max_loc+low_freq_loc+peak_shift])) # comment out if single note
            # peak_data[max_loc+low_freq_loc-peak_shift:max_loc+low_freq_loc+peak_shift] = np.repeat(0,peak_shift*2) # comment out if single note

    freq = max(set(freqs), key=freqs.count)
    print(f"freq {freq}")
    if freq > 100 and freq < 690:
        ref.set({
            'note': freq,
            'note0': freqs[0],
            'note1': freqs[1],
            'note2': freqs[2],
            'note3': freqs[3],
            'note4': freqs[4],
            'note5': freqs[5]
        })

    print()
    plt.pause(0.001)
    # plt.pause(0.001)
    # wait for user to okay the next loop (comment out to have continuous loop)
    # imp = input("Input 0 to Continue, or 1 to save figure ")
    # if imp=='1':
    #     file_name = input("Please input filename for figure ")
    #     plt.savefig(file_name+'.png',dpi=300,facecolor='#FCFCFC')


