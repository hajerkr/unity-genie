import moviepy.editor as mp
import os
import sys

def convert_gif_to_mp4(gif_path):

    clip = mp.VideoFileClip((gif_path))
    video_path = os.path.splitext(f)[0]+'.mp4'
    clip.write_videofile(video_path)
    
    print(f"Converted {f} to {video_path}")
    return video_path
    # if len(sys.argv) == 1:
    #     #return help
    #     print("Please specify filename(s) to convert")
    #     exit(0)
    # else: 
    #     for f in sys.argv[1:]:
            
    