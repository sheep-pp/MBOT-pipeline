# MBOT-pipeline
Multivariate analysis pipeline to assess mice upper limb recovery after SCI

This pipeline enables the analysis of **force data** which is the upper graph, the **position** which is the graph below and the **EMG data** when they are available. 
Through this interactive pipeline, you will be able to correct the peak detection algorithm which is not perfectly suited for noisy, unconsistent signals. 
Once the modifications are made, a json file containing more than 20 metrics is computed and will allow you to perform a PCA analysis. 

**To add points:**
Press 'a' to add a peak (red)
Press 'v' to add a peak onset (green)
Press 'n' to add a peak offset (blue)

**To delete points:**
Press 'z' to delete a peak (red)
Press 'g' to delete a peak onset (green)
Press 'b' to delete a peak offset (blue)

To **move points** position, drap and drop them. 

When you delete a point (a peak, an onset or an offset) always delete their corresponding points. 
When many points are overlapping, it is better t

It is recommanded to check the number of peaks, onset and offset after modification. They should be always at the same number
To do so, press 'p' and check the result on the terminal. 

Press 'q' to quit the video
