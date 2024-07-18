Data presented in Figure 3A-D: "Sensorimotor pathway controlling stopping behavior during chemotaxis in the Drosophila melanogaster larva", Tastekin et al., 2018.

Each file has motorData and sensoryData, which contain kinematic and sensory variables for individual larvae of the experimental groups indicated in file titles.

Single larva tracking (SOS, Gomez-Marin et al., 2012) was used to acquire the data.


|Variables motor|     |
|---------    |---|
|fs| 5|
|scale| scaling factor|
|sourceXY| position of the door source|
|goodFrames| frames in which segmentation was successful|
|lengthSkel| skeleton length|
|area| area of the animal|
|perimeter| perimeter of the animal|
|cmXY| centroid position|
|headXY| head position|
|tailXY| tail position|
|midXY| mid point position|
|cmSpeed| centroid speed|
|headSpeed| head speed|
|tailSpeed| tail speed|
|midSpeed| mid point speed|
|bodyTheta| body angle|
|bodyOmega| body angle speed|
|headTheta| head angle |
|headOmega| head angle speed|
|idxCastEvent| indices for head casts|
|idxTurnStart| indices for beginnings of turns|
|idxTurnEnd| indices for ends of turns|

|Variables sonsory|     |
|---------    |---|
|headSens| odor concentration at the head point|
|cmSens| odor concentration at the centroid |
|tailSens| odor concentration at the tail point|
|midSens| odor concentration at the mid point|
|headSensDot| derivative of odor concentration at the head point|
|cmSensDot| derivative of odor concentration at the centroid|
|tailSensDot| derivative of odor concentration at the tail point|
|midSensDot| derivative of odor concentration at the mid point|
|headSensDotNorm| derivative of odor concentration at the head point (normalized)|
|cmSensDotNorm| derivative of odor concentration at the centroid (normalized)|
|tailSensDotNorm| derivative of odor concentration at the tail point (normalized)|
|midSensDotNorm| derivative of odor concentration at the mid point (normalized)|
|headGrad| gradient at the head point|
|cmGrad| gradient at the centroid|
|tailGrad| gradient at the tail point|
|midGrad| gradient at the mid point|
|bearing| bearing to the odor source calculated using the gradient vectors.|
