# CMEvelocity
 
A novel method for analysis of CME kinematics using Computer vision (opencv2) in Python to recognise and calculate the distance of the front of the CME from the solar centre. This can then be used to find the velocity over time. Steps for this analysis are as follows:
- Convert to Grayscale
- 2 x Median Blur, n = 5
- 3 x Gaussian Blur, n = 5
- Thresholding (values between 200-255)
- Contour Finding
- Distance between furthest point in contour and solar centre

## Example

![solarmaxima](https://user-images.githubusercontent.com/20016090/161012837-c7916644-fe84-4d40-9950-0a800718a614.png)

The graphs above show the results of this method for a CME occuring on 20th September 2017 between 14:00 and 18:00, which gave an average velocity of 1222 m/s. This is similar to the velocity given by the SOHO/LASCO catalog.
