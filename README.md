# Quantification of organoid-images

## Colocalization of diffrent markers and their percentual colocalization
Threshold four ``.tif``-files per organoid. 
The thresholding method will influence the results. Triangle can be too soft, otsu too harsh. Just play aruond with the script.
If it's too soft, there will be large areas with many pixels with low intensities, but that means there are also large areas with a present markers, so the overlap between markers will be influenced. FInd the balance between filtering noise out, and keeping all the important information in the data.
Obtain a lot of values from the images and compare them by all cell lines within each folder of a condition
Compare also the cell lines over different treatments/conditions. Those different treatments need to be stored in seperate folders. 
