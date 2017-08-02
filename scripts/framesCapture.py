import cv2

# ------------- Cheat Sheet -------------
# GOPR0343 - seq12
# GOPR0393 - seq11
# GOPR0395 - seq10
# GOPR0474 - seq01
# GOPR0478 - seq09
# GOPR0484 - seq08
# GOPR0485 - seq07
# GP010343 - seq06
# GP010393 - seq05
# GP010395 - seq04
# GP010474 - seq01
# GP010478 - seq03
# GP020393 - seq02

video_name = "GP010478.mp4"
video_prefix = "seq03/"

video_path_prefix = "/Volumes/Elements/Videos/vids/"
frames_train_path_prefix = "/Volumes/Elements/Videos/frames/train/"+video_prefix
frames_test_path_prefix = "/Volumes/Elements/Videos/frames/test/"+video_prefix

video_path = video_path_prefix + video_name

vidcap = cv2.VideoCapture(video_path)
vidcap.set(cv2.CAP_PROP_POS_MSEC,5000)      # just cue to 5 sec. position

itr = 0
count_train = 1
count_test = 1
flag = False

print "Starting iteration over video ",video_path
while True:
    success,image = vidcap.read()
    if (itr%200==0):
        print "Iteration number: ",itr
        if (itr%1000==0):
            flag = True
        if success:
            if flag:
                path = frames_test_path_prefix + video_prefix
                print count_test,path,success
                cv2.imwrite(path+"_frame%d.jpg" % count_test, image)
                # cv2.waitKey()
                count_test = count_test + 1
            else:
                path = frames_train_path_prefix+video_prefix
                print count_train,path,success
                cv2.imwrite(path+"_frame%d.jpg" % count_train, image)
                # cv2.waitKey()
                count_train = count_train + 1
        else:
            break
    itr = itr + 1
    flag = False
    # cv2.waitKey()
print "Done....."
