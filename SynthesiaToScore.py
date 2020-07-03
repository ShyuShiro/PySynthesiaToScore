import cv2
import os
import numpy as np
from collections import Counter
import pandas as pd
from IPython.display import Image
import mingus.core.value as value
from mingus.containers import NoteContainer
from mingus.containers import Bar
from mingus.containers import Track
from mingus.containers import Composition
from mingus.containers.instrument import Instrument, Piano
from mingus.extra import lilypond as LilyPond
from mingus.midi import midi_file_out as MidiFileOut

###Hyperparameters
debug = False #For output from Pixel <-> Note dictionaries
vod_name = "Above the Treetops - Lith Harbor Synthesia.mp4" #Name of the file to read (.mp4!)
fps = 1 #Speed of the video processing --- Inverse btw .. lower is faster!
measure_length = 57.2 #The number of frames that yield a full measure
start = 230 #Defining the height of the crop region
keysig = "D" #Defining the key signature (in Major only!)
bpm = 120 #Defining the BPM for the midi file -- Has no impact to png/pdf generation
title = "Above the Treetops - Lith Harbor" #Title displayed above the score
author = "Leegle" #Author displayed at the top right of the score
outputname = "LithHarbor" #Name to use for all the output files ... eg: LithHarbor.png/.mid/.pdf
#Mask filter parameters
blu_lower = np.array([75,20,120]) #Lower bound of the Left hand
blu_upper = np.array([140,230,250]) #Upper bound of the Left hand
grn_lower = np.array([30,90,100]) #Lower bound of the Right hand
grn_upper = np.array([80,255,255]) #Upper bound of the Right hand

#Video info constants
cap = cv2.VideoCapture(vod_name) #Load the video
w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH )) #Grab the width of the video
h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT )) #_______ height of the video
vod_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("width: ",w) #640
print("height: ",h) #360
print("FPS: ",vod_fps) #30
print("Video Frame Count: ",total_frames) #8476

if debug==True:    
    print("\nFrame | Note | x-pixel-val\n--------------------------") #Display just for neat output

#Initialize lists to contain the frame + notes that occur on the frame    
frames_left = [0,0] #Initialize the lists with a dummy value
frames_right = [0,0] #Initialize the lists with a dummy value
notes_left = [0] #Initialize the lists with a dummy value
notes_right = [0] #Initialize the lists with a dummy value

#Initialize these locks
allow_duplicate_right = True
allow_duplicate_left = True

#Start image processing ------------------
while True:
    ret, frame = cap.read() #Read 1 frame of video
    
    if frame is None: #If the frame being read is None (ie: End of video)
        break #Exit app
    
    #Crop image to only 1 pixel tall image
    stop = start+1
    cropped = frame[start:stop, 0:w]
    
    #Create a visual indicator of this cropped region -- red rectangle above the keyboard
    box = cv2.rectangle(frame, #What object to draw the rectangle on
                  (0,start-1), #Start drawing (Top left corner)
                  (w,stop), #End drawing (Bottom right corner)
                  (0,0,255), #BGR color code 
                  1) #line width, -1 fills it instead of line width

    #Convert image to HSV (Hue, Saturation, Value) instead of BGR for better control of green vs blue colors
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    hsv_crop = cv2.cvtColor(cropped,cv2.COLOR_BGR2HSV) #Convert crop image to hsv scale
    
    #Blue filter
    blu_mask = cv2.inRange(hsv,blu_lower,blu_upper)
    blu_mask_crop = cv2.inRange(hsv_crop,blu_lower,blu_upper)
    blu_filtered = cv2.bitwise_and(frame,frame,mask=blu_mask) #Find in "Frame" then show pixels of "Frame" where "mask = 1"
    blu_cropped = cv2.bitwise_and(cropped,cropped,mask=blu_mask_crop) #Find only blue pixels in the cropped image
    
    #Green filter
    grn_mask = cv2.inRange(hsv,grn_lower,grn_upper)
    grn_mask_crop = cv2.inRange(hsv_crop,grn_lower,grn_upper)
    grn_filtered = cv2.bitwise_and(frame,frame,mask=grn_mask)
    grn_cropped = cv2.bitwise_and(cropped,cropped,mask=grn_mask_crop)
    
    cv2.imshow("Original",frame) #Display original video    
    cv2.imshow("Blue",blu_filtered)
    cv2.imshow("Green",grn_filtered)
    #cv2.imshow("Cropped",cropped)
    #cv2.imshow("Blue_cropped",blu_cropped)
    #cv2.imshow("Green_cropped",grn_cropped)

    current_frame = int(cap.get(1)) #Current frame_number of the video
    
    #Note detection
    if current_frame%2 == 0: #Read every even frame (to mitigate pixel bleed from 1 frame to next)
            
        coord_right=cv2.findNonZero(grn_mask_crop) #Pixel locations of where there are green
        coord_left=cv2.findNonZero(blu_mask_crop) #Pixel locations of where there are blue
            
        #Right hand detection
        if type(coord_right) == np.ndarray: #If not None basically
            note_right = ConvertListToNote(coord_right[:,0][:,0]) #Convert the coordinates to a note
            #Update only if there is a change to previously detected notes
            #Issue-- What if its a repeating note such as a baseline (common on left hand)
            #Solution 6/23: If the last frame entry is the next frame (+2) then likely a sustained note
            #Solution 6/25: frame condition is redundant (already processing every other frame)
                #Should instead determine if coord_right has read a "None" detection
                #If so, then create a boolean lock to allow duplicate notes
                #None detection denotes that there was a gap (ex: repeated bass notes)
            if note_right != None:
                if note_right != notes_right[-1] or allow_duplicate_right==True: 
                    #Register frames & notes to appropriate lists
                    if debug==True:
                        print("RH:",current_frame,note_right)
                    if len(note_right) > 0: #If the note returned is not empty
                        #Add to registry
                        frames_right.append(current_frame)
                        notes_right.append(note_right)
                        #print("RH:",current_frame,note_right)
                        allow_duplicate_right = False #Disallow duplication of this held note
        else: #If previous reading was "None" (the only other reading type) allow duplicate notes
            allow_duplicate_right = True
                
        #Left hand detection
        if type(coord_left) == np.ndarray: #If not None basically
            note_left = ConvertListToNote(coord_left[:,0][:,0]) #Convert the coordinates to a note
            #Update only if there is a change to previously detected notes
            if note_left != None:
                if note_left != notes_left[-1] or allow_duplicate_left==True: 
                    #Register frames & notes to appropriate lists
                    if debug==True:
                        print("LH",current_frame,note_left)
                    if len(note_left) > 0: #If the note returned is not empty
                        #Add to registry
                        frames_left.append(current_frame)
                        notes_left.append(note_left)                
                        allow_duplicate_left = False #Disallow duplication of this held note
        else: #If previous reading was "None" (the only other reading type) allow duplicate notes
            allow_duplicate_left = True
            
    #Block to allow breakage of code without Ctrl+C
    if cv2.waitKey(fps) & 0xFF== ord('q'): #waitKey(fps) controls the video speed
        break #If "q" is pushed, close the app

#Close the video feed
cap.release() #Unhook the video -- otherwise the next run of the app won't be able to access it
cv2.destroyAllWindows() #Close all windows generated by cv2 commands

#Clean up the frames and notes lists for both left and right hand
#They were first initialized with value "0" as their first entry (first 2 entries for frames)
frames_left = frames_left[2:]
frames_right = frames_right[2:]
notes_left = notes_left[1:]
notes_right = notes_right[1:]

#Although the video starts at frame 0 -- the first note does not
#Standardize by finding the first note instance
first_frame = min(frames_left[0],frames_right[0])
normalized_left = [np.round((x-first_frame)/measure_length,2) for x in frames_left] #Subtract 'first_frame' from each entry
normalized_right = [np.round((x-first_frame)/measure_length,2) for x in frames_right] #and convert to measures by dividing by measure_length

#LilyPond -- Convert {Frames,Notes} pairs into piano sheet music
normalized = normalized_left + normalized_right
all_notes = notes_left+notes_right

#Create left and right hand frame/note pairs for debug reading
left = list(zip(frames_left,normalized_left,notes_left))
right = list(zip(frames_right,normalized_right,notes_right))

#Create PD dataframe to sort notes & compute measures
df_left = pd.DataFrame({"Frame":normalized_left,
                   "Notes_l":notes_left,
                   })
df_right = pd.DataFrame({"Frame":normalized_right,
                   "Notes_r":notes_right,
                        })

df = pd.merge(df_left, df_right, how='outer', on=['Frame']) #Combine left & right dataframes
df = df.sort_values("Frame").reset_index(drop=True) #sort by frame value and reset index

##Left hand Computation
#Compute note lengths via Offseting the time value
offset = list(df["Frame"].values)[1:]
if df["Frame"].iloc[-1] == np.round(df["Frame"].iloc[-1],0):
    offset.append(df["Frame"].iloc[-1]+1)
else:
    offset.append(np.ceil(df["Frame"].iloc[-1]))    

#Compute note length & add as column
#--- Round the difference values to nearest 0.0625 (sixteenth note) value
diff = [np.round((y-x)/0.0625)*0.0625 for x,y in zip(list(df["Frame"].values),offset)]
df["Diff"] = diff

#Create "Measures" column by taking a ceiling function
#Apply a small perturbation by a factor much smaller than the smallest unit
#so that values such as 0.00 become 1 and 1.00 become 2.
df["Measure"] = (df["Frame"]+0.0001).apply(np.ceil)

#Change all NaN (should be from the Diff column) to blanks
df = df.fillna('')

#Last step -- Dictionary convert note length to note type (eigth/quarter etc)
LengthToNote = dict({0.0625:16,
                     0.125:8,
                     0.1875:8,
                     0.25:4,
                     0.3125:4, #Round to dotted half
                     0.375:value.dots(4), #Dotted quarter
                     0.4375:value.dots(4), #Round to dotted half
                     0.5:2,
                     0.5625:2, #Round to half
                     0.625:2, #Round to half
                     0.6875:2, #Round to half
                     0.75:value.dots(2), #Dotted half
                     0.8125:value.dots(2), #Round to dotted half
                     0.875:value.dots(2), #Round to dotted half
                     0.9375:value.dots(2), #Round to dotted half                
                     1:1})
df["Diff"] = df["Diff"].map(LengthToNote) #Map to note values for Mingus

display(df.head(10))

#Creating tracks --- Left
t_l = Track()
for i in set(df["Measure"].unique()[:-1]):
    b = Bar(key=keysig)
    subset = df[df["Measure"]==i]
    for j,k in zip(subset["Notes_l"],subset["Diff"]):
        if len(j)>0: #If note is not NaN
            nc = NoteContainer(j) #Define the note
            b.place_notes(nc,k) #Add note to bar with length
        else:
            b.place_notes(None,k) #Place a rest note the lenght of the other hand's note
    t_l + b
LithHarbor_ly_left = LilyPond.from_Track(t_l) #Create .ly file
    
#Creating tracks --- right
t_r = Track()
for i in set(df["Measure"].unique()[:-1]):
    b = Bar(key=keysig)
    subset = df[df["Measure"]==i]
    for j,k in zip(subset["Notes_r"],subset["Diff"]):
        if j: #If note is not NaN
            nc = NoteContainer(j) #Define the note
            b.place_notes(nc,k) #Add note to bar with length
        else:
            b.place_notes(None,k) #Place a rest note the lenght of the other hand's note
    t_r + b
LithHarbor_ly_right = LilyPond.from_Track(t_r) #Create .ly file

#Remove the old png file so that the code doesn't load an old instance of the file
try:
    os.remove(outputname+".png")
    os.remove(outputname+".mid")
except:
    pass

#### Create png/pdf file
#Need to stitch the Left and Right hand .ly strings together
#1) Define header (Title and author of the composition)
#2) Define Right hand (From LithHarbor_ly_right)
#3) Define Left hand (From LithHarbor_ly_left)
#4) Set up piano score and staff with "RH" as right hand and "LH" as left-hand
#5) Create png/pdf and display (if png)
header = '\\header { title = "' + title + '" composer = "' + author + '" opus = "" } '
combine_test = header + "rhMusic =  {" + LithHarbor_ly_right + "}"
combine_test = combine_test + "lhMusic =  {" + LithHarbor_ly_left + "}"
combine_test = combine_test + """
\\score {
  \\new PianoStaff <<
    \\new Staff = "RH"  <<
      \\rhMusic
    >>
    \\new Staff = "LH" <<
      \\clef "bass"
      \\lhMusic
    >>
  >>
}"""

LilyPond.to_png(combine_test, outputname) #Create png

#### Create midi file:
#Combine Notes_r and Notes_l from the df into 1 congomerate
combined_notes = []
for i in range(df.shape[0]):
    try:
        combined_notes.append(df.iloc[i]["Notes_l"]+df.iloc[i]["Notes_r"])
    except:
        if df.iloc[i]["Notes_l"]!="":
            combined_notes.append(df.iloc[i]["Notes_l"])
        else:
            combined_notes.append(df.iloc[i]["Notes_r"])
df["Notes_both"] = combined_notes

#Now create a track by adding these notes from both hands
t_both = Track()
for i in set(df["Measure"].unique()[:-1]):
    b = Bar(key=keysig)
    subset = df[df["Measure"]==i]
    for j,k in zip(subset["Notes_both"],subset["Diff"]):
        if j: #If note is not NaN
            nc = NoteContainer(j) #Define the note
            b.place_notes(nc,k) #Add note to bar with length
    t_both + b
MidiFileOut.write_Track(outputname+".mid",t_both,bpm=bpm) #Create midi

#Display first page of composition
try:
    Image(filename=outputname+".png")
except:
    Image(filename=outputname+"-page1.png")
