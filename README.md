# PySynthesiaToScore 7/3/2020 -- Brandon Tran (ShyuShiro)
Generating musical compositions from Synthesia mp4 videos

Hi! This was a passion project birthed from my desire to play a song from my childhood named "Above the Treetops" from Maplestory.

---- What is this project? In short:

Input = .mp4 video of a Synthesia piano
Output = Musical composition of that Synthesia video


---- What is this project? In detail:
The script processes a .mp4 file through OpenCV library to detect pixels falling through a detection bar.

These pixels are converted to actual notes through a dictionary (which required me to literally count "Pixel 340 = G-3" ... you're welcome)

Note durations are computed based on a difference to the next note being played --- through difference in the frame at which both these notes get played.

Now that you have Note ID and Note Duration, Mingus is used to stitch the notes into a Bar object (musical measure) and put onto a Track (musical score).

Lastly Mingus communicates with LilyPond to generate a png/pdf/midi output file of the musical composition.

---- Requirements:
1) Standard Python packages --- pandas, cv2, numpy
2) mingus --- See https://pypi.org/project/mingus/ on how to download & use
3) LilyPond software --- See http://lilypond.org/ on how to download AND INSTALL TO YOUR PATH ENVIRONMENT (otherwise mingus cant execute it)


---- How to use:
I recommend using the jupyter notebook file, but that is my preference --- Because I didn't build a class structure the .py file is NOT standalone.
The .py file is provided simply in case you want to upload this into your own preference of IDE (Ex: Spyder or PyCharm).

1) Run the code block
2) Terminate the script early with "q" or simply let it completly analyze the entire video.

I say "terminate early" because some videos have a "Thank you" fade out at the end of the synthesia video. (or many other after effects)
If you don't want nonsense notes being recorded ... simply terminate the video as this happens.


---- How to tune the parameters for your video:
RECOMMEND SEEING THE HTML FILE AS IT MAY GIVE YOU A BETTER UNDERSTANDING OF HOW TO TUNE THE PARAMETERS

You may need to change:
1) Change the hyperparameters relating to your video
-- Eg: vod_name should match the .mp4 file you want to read ... among other things like "title" and "author"
2) `Start` variable may need to be adjusted higher or lower if there are any edits to the video that may interfere with the pixel detection bar (red bar)
3) measure_length will NEED to be modified. This variable is measuring "How long (in frames) is 1 bar of music in this composition?"

Note that measure_length is not going to be a simple task to change.
You'll need some sense of the musical composition you're analyzing to adjust it, along with some trial & error.
Simply run the script and hit "q" to exit the file after 3~4 measures have played and see if the notes line up. If not, adjust the measure_length variable until they do.

4) If your synthesia video is not the default blue & green colors for Left and Right hands respectively --- Adjust the masking of blu_lower and blu_upper mask values.

For this, you'll need to look into HSV (Hue, Saturation, Value) and it can take a lot of trial & error again as well.

If you need more assistance ... please refer to the html file as I've provided detailed information on how the script works which may aid in your tuning process.
I wish you the best of luck if you are trying to adapt this file to your video!

---- Disclaimers:
1) This is not perfect! It will
- Mis-interpret notes (Eg: and "A" for an "A#")
- Very likely the measures will be wrong (Eg: Notes are truncated or put into the wrong measures)
- As a result of measures being wrong -- Notes will be completely missing from the score (void and gone)

2) On that note -- I've used software/packages such as OpenCV, Mingus, and LilyPond (beyond the basic Python packages of pandas and numpy). 
Please respect their license agreements when using mine or any of their scripts/products!

3) I'm releasing this code as a "please help me improve" gesture as I'm only a hobby programmer at the moment! Please give me input and help me improve as a programmer.
