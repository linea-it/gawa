
README for GAWA VERSION2 

gawa code is composed of 

3 call scripts : 
- gawa_multithread_call.py 
- gawa_thread_call.py 
- gawa_concatenate.py 

3 libraries : 
- gawa.py
- utils.py 
- multithread.py 

1 config file 
- gawa.cfg 

Launch gawa in one line : 
> python gawa_multithread_call.py 

Note that : 
- In this disctributed version gawa_multithread_call.py  
 calls gawa_thread_call.py in a LOOP. This part has to 
 be substituted by your favourite batch scheduler. 
 Also the 2nd argument of gawa_thread_call.py (the thread id)
 should become the batch scheduler local id. 

- gawa_multithread_call.py runs the fcts: 
  + gawa/compute_dslices
  + gawa/compute_cmd_masks
 The first fct defines the list of distances (slices) 
 based on some dist_min, dist_max in the .cfg 
 and also assuming some overlap of the mask to assure that there
 is no "hole" in distance for detection. 
 These functions may be improved once we define clearly our 
 strategy which is not clear at the time this code is delivered. 


What is new in GAWA VERSION2 :
- better modularity of the code 
- several functions were factorized 
- the tiling is done in healpix pixels so that 
  it can operate at any RA-Dec
- the divisision of the N tiles in P cores is 
  done to optimize the distribution of the area 
  to be analyzed (as equal area as possible). 
- SNR has a new definition
- cylinders are kept 
- identification of clusters in cylinders is performed
- method for filtering of possible duplicates has improved 
  in the tiles and between the tiles. 
- several steps of the codes were rewritten to decrease 
  CPU time. Some steps were improved by a factor > 10. 
- there are 3 levels of verbose. With verbose = 0 no intermediate 
  file is written on disc except those necessary for the code.
  (problem of SPARSE2D - see below).
- there are several re-entry points with the generation of 
  numpy files (.npx). But this can be switched off if necessary


What should be improved : 
- currently the code provides detections at discrete distances 
  corresponding to those provided by the gawa/compute_dslices function.
  We should add a fct to refine this first distance estimate. 
- Update the SPARSE2D package which is currently a C++ code => python 
  version to avoid current i/o's 
- ..

