1

   Abaqus 3DEXPERIENCE R2018x                                  Date 28-Jan-2022   Time 11:31:15
   For use by UNIVERSITY COLLEGE LONDON under license from Dassault Systemes or its subsidiary.

                                                                                 
                                                                                 
 STEP    1     INCREMENT     1     STEP TIME    0.00    


                        S T E P       1     S T A T I C   A N A L Y S I S


                                                                                          

     AUTOMATIC TIME CONTROL WITH -
          A SUGGESTED INITIAL TIME INCREMENT OF                1.000E-02
          AND A TOTAL TIME PERIOD OF                            1.00    
          THE MINIMUM TIME INCREMENT ALLOWED IS                1.000E-02
          THE MAXIMUM TIME INCREMENT ALLOWED IS                1.000E-02

     LINEAR EQUATION SOLVER TYPE         DIRECT SPARSE

 CONVERGENCE TOLERANCE PARAMETERS FOR FORCE    
     CRITERION FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR DISP.    CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INITIAL VALUE OF TIME AVERAGE FORCE                               1.000E-02
     AVERAGE FORCE     IS TIME AVERAGE FORCE    
     ALTERNATE CRIT. FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO FORCE     RELATIVE TO TIME AVRG. FORCE         1.000E-05
     CRITERION FOR RESIDUAL FORCE     WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR DISP.    CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL FORCE     FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO FORCE     REL. TO TIME AVRG. MAX. FORCE        1.000E-05
     CRITERION FOR ZERO DISP.    RELATIVE TO CHARACTERISTIC LENGTH     1.000E-08

 CONVERGENCE TOLERANCE PARAMETERS FOR MOMENT   
     CRITERION FOR RESIDUAL MOMENT    FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR ROTATION CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INITIAL VALUE OF TIME AVERAGE MOMENT                              1.000E-02
     AVERAGE MOMENT    IS TIME AVERAGE MOMENT   
     ALTERNATE CRIT. FOR RESIDUAL MOMENT    FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO MOMENT    RELATIVE TO TIME AVRG. MOMENT        1.000E-05
     CRITERION FOR RESIDUAL MOMENT    WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR ROTATION CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL MOMENT    FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO MOMENT    REL. TO TIME AVRG. MAX. MOMENT       1.000E-05

     VOLUMETRIC STRAIN COMPATIBILITY TOLERANCE FOR HYBRID SOLIDS       1.000E-05
     AXIAL STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS             1.000E-05
     TRANS. SHEAR STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS      1.000E-05
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P>P0          5.000E-03
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P=0.0         0.100    
     CONTACT FORCE ERROR TOLERANCE FOR CONVERT SDI=YES                 1.00    
     DISPLACEMENT COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS           1.000E-05
     ROTATION COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS               1.000E-05

 EQUILIBRIUM WILL BE CHECKED FOR SEVERE DISCONTINUITY ITERATIONS

 TIME INCREMENTATION CONTROL PARAMETERS:
     FIRST EQUILIBRIUM ITERATION FOR CONSECUTIVE DIVERGENCE CHECK              4
     EQUILIBRIUM ITERATION AT WHICH LOG. CONVERGENCE RATE CHECK BEGINS         8
     EQUILIBRIUM ITERATION AFTER WHICH ALTERNATE RESIDUAL IS USED              9
     MAXIMUM EQUILIBRIUM ITERATIONS ALLOWED                                   16
     EQUILIBRIUM ITERATION COUNT FOR CUT-BACK IN NEXT INCREMENT               10
     MAXIMUM EQUILIB. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      4
     MAXIMUM ITERATIONS FOR SEVERE DISCONTINUITIES                            50
     MAXIMUM ATTEMPTS ALLOWED IN AN INCREMENT                                  5
     MAXIMUM DISCON. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      50
     MAXIMUM CONTACT AUGMENTATIONS FOR *SURFACE BEHAVIOR,AUGMENTED LAGRANGE   50
     CUT-BACK FACTOR AFTER DIVERGENCE                                 0.2500    
     CUT-BACK FACTOR FOR TOO SLOW CONVERGENCE                         0.5000    
     CUT-BACK FACTOR AFTER TOO MANY EQUILIBRIUM ITERATIONS            0.7500    
     CUT-BACK FACTOR AFTER TOO MANY SEVERE DISCONTINUITY ITERATIONS   0.2500    
     CUT-BACK FACTOR AFTER PROBLEMS IN ELEMENT ASSEMBLY               0.2500    
     INCREASE FACTOR AFTER TWO INCREMENTS THAT CONVERGE QUICKLY        1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED                       1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DYNAMICS)            1.250    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DIFFUSION)           2.000    
     MINIMUM TIME INCREMENT RATIO FOR EXTRAPOLATION TO OCCUR          0.1000    
     MAX. RATIO OF TIME INCREMENT TO STABILITY LIMIT                   1.000    
     FRACTION OF STABILITY LIMIT FOR NEW TIME INCREMENT               0.9500    
     TIME INCREMENT INCREASE FACTOR BEFORE A TIME POINT                1.000    
     GLOBAL STABILIZATION CONTROL IS NOT USED

          PRINT OF INCREMENT NUMBER, TIME, ETC., EVERY    1  INCREMENTS

     THE MAXIMUM NUMBER OF INCREMENTS IN THIS STEP IS                    1000

     LINEAR EXTRAPOLATION WILL BE USED

     CHARACTERISTIC ELEMENT LENGTH     1.000E-02

     DETAILS REGARDING ACTUAL SOLUTION WAVEFRONT REQUESTED

     DETAILED OUTPUT OF DIAGNOSTICS TO DATABASE REQUESTED

     PRINT OF INCREMENT NUMBER, TIME, ETC., TO THE MESSAGE FILE EVERY     1  INCREMENTS

     COLLECTING MODEL CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS

     COLLECTING STEP CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS


  INCREMENT     1 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 33.5296E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 146.439E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.067E-03   TIME AVG. FORCE       8.067E-03
 LARGEST RESIDUAL FORCE            -1.397E-13   AT NODE         61   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     1.747E-03   TIME AVG. MOMENT      1.747E-03
 LARGEST RESIDUAL MOMENT           -4.658E-16   AT NODE        548   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.023E-03   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -1.023E-03   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  1.000E-02
 STEP TIME COMPLETED       1.000E-02,  TOTAL TIME COMPLETED        1.000E-02


  INCREMENT     2 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.227E-02   TIME AVG. FORCE       2.017E-02
 LARGEST RESIDUAL FORCE            -8.372E-14   AT NODE        213   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -7.464E-15   AT NODE        134   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.293E-03   TIME AVG. MOMENT      3.520E-03
 LARGEST RESIDUAL MOMENT            4.372E-16   AT NODE        128   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.023E-03   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.588E-14   AT NODE        473   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  2.000E-02
 STEP TIME COMPLETED       2.000E-02,  TOTAL TIME COMPLETED        2.000E-02


  INCREMENT     3 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.840E-02   TIME AVG. FORCE       2.958E-02
 LARGEST RESIDUAL FORCE             1.097E-13   AT NODE        118   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.733E-15   AT NODE        134   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.939E-03   TIME AVG. MOMENT      4.993E-03
 LARGEST RESIDUAL MOMENT           -4.129E-16   AT NODE        392   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.023E-03   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -7.945E-15   AT NODE        473   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  3.000E-02
 STEP TIME COMPLETED       3.000E-02,  TOTAL TIME COMPLETED        3.000E-02


  INCREMENT     4 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.454E-02   TIME AVG. FORCE       3.832E-02
 LARGEST RESIDUAL FORCE            -1.682E-13   AT NODE        387   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.629E-18   AT NODE        160   DOF  1
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     1.059E-02   TIME AVG. MOMENT      6.391E-03
 LARGEST RESIDUAL MOMENT            8.066E-16   AT NODE        123   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.023E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     3.869E-18   AT NODE        521   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  4.000E-02
 STEP TIME COMPLETED       4.000E-02,  TOTAL TIME COMPLETED        4.000E-02


  INCREMENT     5 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.067E-02   TIME AVG. FORCE       4.679E-02
 LARGEST RESIDUAL FORCE            -2.380E-13   AT NODE        388   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        5.035E-18   AT NODE        119   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     1.323E-02   TIME AVG. MOMENT      7.759E-03
 LARGEST RESIDUAL MOMENT           -7.910E-16   AT NODE        377   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.023E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     5.997E-18   AT NODE        804   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  5.000E-02
 STEP TIME COMPLETED       5.000E-02,  TOTAL TIME COMPLETED        5.000E-02


  INCREMENT     6 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.681E-02   TIME AVG. FORCE       5.513E-02
 LARGEST RESIDUAL FORCE            -2.098E-13   AT NODE         12   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -4.160E-18   AT NODE        137   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     1.588E-02   TIME AVG. MOMENT      9.112E-03
 LARGEST RESIDUAL MOMENT           -6.800E-16   AT NODE        104   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.023E-03   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     7.261E-18   AT NODE        832   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  6.000E-02
 STEP TIME COMPLETED       6.000E-02,  TOTAL TIME COMPLETED        6.000E-02


  INCREMENT     7 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.113       TIME AVG. FORCE       6.339E-02
 LARGEST RESIDUAL FORCE            -3.171E-13   AT NODE        555   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        4.632E-18   AT NODE        103   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     1.852E-02   TIME AVG. MOMENT      1.046E-02
 LARGEST RESIDUAL MOMENT           -1.433E-15   AT NODE        134   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.023E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -9.874E-18   AT NODE        890   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  7.000E-02
 STEP TIME COMPLETED       7.000E-02,  TOTAL TIME COMPLETED        7.000E-02


  INCREMENT     8 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.129       TIME AVG. FORCE       7.160E-02
 LARGEST RESIDUAL FORCE            -3.296E-13   AT NODE        116   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -6.246E-18   AT NODE        557   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     2.117E-02   TIME AVG. MOMENT      1.180E-02
 LARGEST RESIDUAL MOMENT           -1.392E-15   AT NODE        208   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.023E-03   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -1.152E-17   AT NODE         90   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  8.000E-02
 STEP TIME COMPLETED       8.000E-02,  TOTAL TIME COMPLETED        8.000E-02


  INCREMENT     9 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.145       TIME AVG. FORCE       7.978E-02
 LARGEST RESIDUAL FORCE             2.888E-13   AT NODE        117   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -9.223E-18   AT NODE        146   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     2.382E-02   TIME AVG. MOMENT      1.313E-02
 LARGEST RESIDUAL MOMENT            1.527E-15   AT NODE        377   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.023E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -1.335E-17   AT NODE        192   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  9.000E-02
 STEP TIME COMPLETED       9.000E-02,  TOTAL TIME COMPLETED        9.000E-02


  INCREMENT    10 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.161       TIME AVG. FORCE       8.793E-02
 LARGEST RESIDUAL FORCE            -2.974E-13   AT NODE         84   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -8.786E-18   AT NODE        595   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     2.646E-02   TIME AVG. MOMENT      1.447E-02
 LARGEST RESIDUAL MOMENT            1.610E-15   AT NODE        392   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.023E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.507E-17   AT NODE        615   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.100    
 STEP TIME COMPLETED       0.100    ,  TOTAL TIME COMPLETED        0.100    


  INCREMENT    11 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.177       TIME AVG. FORCE       9.607E-02
 LARGEST RESIDUAL FORCE            -3.982E-13   AT NODE         87   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.154E-17   AT NODE        252   DOF  1
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     2.911E-02   TIME AVG. MOMENT      1.580E-02
 LARGEST RESIDUAL MOMENT            1.645E-15   AT NODE        126   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.023E-03   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -1.519E-17   AT NODE         20   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.110    
 STEP TIME COMPLETED       0.110    ,  TOTAL TIME COMPLETED        0.110    


  INCREMENT    12 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.194       TIME AVG. FORCE       0.104    
 LARGEST RESIDUAL FORCE            -3.717E-13   AT NODE         13   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        8.568E-18   AT NODE        471   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.176E-02   TIME AVG. MOMENT      1.713E-02
 LARGEST RESIDUAL MOMENT           -1.936E-15   AT NODE        377   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.023E-03   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.492E-17   AT NODE        167   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.120    
 STEP TIME COMPLETED       0.120    ,  TOTAL TIME COMPLETED        0.120    


  INCREMENT    13 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.210       TIME AVG. FORCE       0.112    
 LARGEST RESIDUAL FORCE            -3.530E-13   AT NODE        119   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        8.902E-18   AT NODE        278   DOF  1
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.440E-02   TIME AVG. MOMENT      1.846E-02
 LARGEST RESIDUAL MOMENT            1.960E-15   AT NODE        576   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.023E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -9.570E-18   AT NODE        611   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.130    
 STEP TIME COMPLETED       0.130    ,  TOTAL TIME COMPLETED        0.130    


  INCREMENT    14 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.226       TIME AVG. FORCE       0.120    
 LARGEST RESIDUAL FORCE             5.512E-13   AT NODE        112   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.017E-17   AT NODE        362   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.705E-02   TIME AVG. MOMENT      1.978E-02
 LARGEST RESIDUAL MOMENT           -3.025E-15   AT NODE        107   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.023E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.278E-17   AT NODE        638   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.140    
 STEP TIME COMPLETED       0.140    ,  TOTAL TIME COMPLETED        0.140    


  INCREMENT    15 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.242       TIME AVG. FORCE       0.129    
 LARGEST RESIDUAL FORCE            -5.174E-13   AT NODE        561   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.559E-17   AT NODE        378   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.970E-02   TIME AVG. MOMENT      2.111E-02
 LARGEST RESIDUAL MOMENT           -3.095E-15   AT NODE        564   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.023E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -2.685E-17   AT NODE        469   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.150    
 STEP TIME COMPLETED       0.150    ,  TOTAL TIME COMPLETED        0.150    


  INCREMENT    16 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.258       TIME AVG. FORCE       0.137    
 LARGEST RESIDUAL FORCE            -5.489E-13   AT NODE        566   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.227E-17   AT NODE        153   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.234E-02   TIME AVG. MOMENT      2.244E-02
 LARGEST RESIDUAL MOMENT            2.921E-15   AT NODE         91   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.023E-03   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     3.111E-17   AT NODE        347   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.160    
 STEP TIME COMPLETED       0.160    ,  TOTAL TIME COMPLETED        0.160    


  INCREMENT    17 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.274       TIME AVG. FORCE       0.145    
 LARGEST RESIDUAL FORCE             6.487E-13   AT NODE        119   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.831E-17   AT NODE         84   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.499E-02   TIME AVG. MOMENT      2.376E-02
 LARGEST RESIDUAL MOMENT           -3.275E-15   AT NODE        120   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.023E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     2.741E-17   AT NODE        705   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.170    
 STEP TIME COMPLETED       0.170    ,  TOTAL TIME COMPLETED        0.170    


  INCREMENT    18 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.287       TIME AVG. FORCE       0.153    
 LARGEST RESIDUAL FORCE            -3.704E-03   AT NODE         16   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -4.868E-07   AT NODE          6   DOF  2
   INSTANCE: INST                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     4.762E-02   TIME AVG. MOMENT      2.509E-02
 LARGEST RESIDUAL MOMENT            8.304E-05   AT NODE         16   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.025E-03   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -4.562E-06   AT NODE        449   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 33.2660E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 138.623E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.290       TIME AVG. FORCE       0.153    
 LARGEST RESIDUAL FORCE            -7.707E-13   AT NODE        120   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.242E-07   AT NODE         16   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     4.761E-02   TIME AVG. MOMENT      2.509E-02
 LARGEST RESIDUAL MOMENT           -2.970E-15   AT NODE        395   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.026E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     2.152E-06   AT NODE        175   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.180    
 STEP TIME COMPLETED       0.180    ,  TOTAL TIME COMPLETED        0.180    


  INCREMENT    19 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 34.5443E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 123.597E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.306       TIME AVG. FORCE       0.161    
 LARGEST RESIDUAL FORCE            -9.357E-13   AT NODE        133   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.872E-06   AT NODE        175   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.016E-02   TIME AVG. MOMENT      2.641E-02
 LARGEST RESIDUAL MOMENT           -3.955E-15   AT NODE        551   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.041E-03   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -2.716E-05   AT NODE        336   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.190    
 STEP TIME COMPLETED       0.190    ,  TOTAL TIME COMPLETED        0.190    


  INCREMENT    20 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.313       TIME AVG. FORCE       0.168    
 LARGEST RESIDUAL FORCE            -9.045E-03   AT NODE        450   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.587E-06   AT NODE        771   DOF  2
   INSTANCE: INST                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     5.265E-02   TIME AVG. MOMENT      2.772E-02
 LARGEST RESIDUAL MOMENT           -1.720E-04   AT NODE        175   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.050E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.473E-05   AT NODE        838   DOF  6
   INSTANCE: INST                                                                            
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 27.4025E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 127.692E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.321       TIME AVG. FORCE       0.169    
 LARGEST RESIDUAL FORCE             7.308E-13   AT NODE        121   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        6.466E-07   AT NODE        336   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     5.264E-02   TIME AVG. MOMENT      2.772E-02
 LARGEST RESIDUAL MOMENT            3.164E-15   AT NODE        109   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.054E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -6.084E-06   AT NODE        451   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.200    
 STEP TIME COMPLETED       0.200    ,  TOTAL TIME COMPLETED        0.200    


  INCREMENT    21 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 24.5028E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 107.755E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.336       TIME AVG. FORCE       0.177    
 LARGEST RESIDUAL FORCE             7.210E-13   AT NODE        393   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.696E-06   AT NODE        450   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.508E-02   TIME AVG. MOMENT      2.902E-02
 LARGEST RESIDUAL MOMENT           -3.358E-15   AT NODE        400   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.065E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.657E-05   AT NODE        579   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.210    
 STEP TIME COMPLETED       0.210    ,  TOTAL TIME COMPLETED        0.210    


  INCREMENT    22 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.337       TIME AVG. FORCE       0.184    
 LARGEST RESIDUAL FORCE             4.578E-03   AT NODE         10   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.021E-06   AT NODE        773   DOF  2
   INSTANCE: INST                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     5.747E-02   TIME AVG. MOMENT      3.032E-02
 LARGEST RESIDUAL MOMENT            1.677E-04   AT NODE        515   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.071E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.124E-05   AT NODE        839   DOF  6
   INSTANCE: INST                                                                            
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 23.0502E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 260.881E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.350       TIME AVG. FORCE       0.185    
 LARGEST RESIDUAL FORCE            -8.635E-13   AT NODE         96   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.019E-07   AT NODE          1   DOF  1
   INSTANCE: {assembly node}                                                                 
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     5.746E-02   TIME AVG. MOMENT      3.032E-02
 LARGEST RESIDUAL MOMENT            4.746E-15   AT NODE         97   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.071E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -2.206E-06   AT NODE        173   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.220    
 STEP TIME COMPLETED       0.220    ,  TOTAL TIME COMPLETED        0.220    


  INCREMENT    23 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 29.6396E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 188.287E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.364       TIME AVG. FORCE       0.193    
 LARGEST RESIDUAL FORCE            -7.530E-13   AT NODE        104   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.667E-06   AT NODE        616   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.968E-02   TIME AVG. MOMENT      3.159E-02
 LARGEST RESIDUAL MOMENT            4.094E-15   AT NODE        108   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.050E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -2.542E-05   AT NODE        306   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.230    
 STEP TIME COMPLETED       0.230    ,  TOTAL TIME COMPLETED        0.230    


  INCREMENT    24 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.362       TIME AVG. FORCE       0.200    
 LARGEST RESIDUAL FORCE             1.323E-02   AT NODE        742   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.846E-06   AT NODE        332   DOF  2
   INSTANCE: INST                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     6.177E-02   TIME AVG. MOMENT      3.285E-02
 LARGEST RESIDUAL MOMENT           -4.125E-04   AT NODE        741   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.035E-03   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.643E-05   AT NODE        202   DOF  6
   INSTANCE: INST                                                                            
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 56.3479E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 241.497E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.377       TIME AVG. FORCE       0.200    
 LARGEST RESIDUAL FORCE             8.867E-13   AT NODE        140   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        7.951E-07   AT NODE        577   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     6.176E-02   TIME AVG. MOMENT      3.285E-02
 LARGEST RESIDUAL MOMENT            4.274E-15   AT NODE        139   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.029E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -6.245E-06   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.240    
 STEP TIME COMPLETED       0.240    ,  TOTAL TIME COMPLETED        0.240    


  INCREMENT    25 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 42.5476E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 299.103E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.389       TIME AVG. FORCE       0.208    
 LARGEST RESIDUAL FORCE             8.485E-13   AT NODE        111   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.649E-06   AT NODE         17   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     6.374E-02   TIME AVG. MOMENT      3.409E-02
 LARGEST RESIDUAL MOMENT           -5.662E-15   AT NODE        116   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.016E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -1.574E-05   AT NODE        489   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.250    
 STEP TIME COMPLETED       0.250    ,  TOTAL TIME COMPLETED        0.250    


  INCREMENT    26 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.380       TIME AVG. FORCE       0.214    
 LARGEST RESIDUAL FORCE             1.313E-02   AT NODE        551   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        8.523E-07   AT NODE          6   DOF  2
   INSTANCE: INST                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     6.564E-02   TIME AVG. MOMENT      3.530E-02
 LARGEST RESIDUAL MOMENT            3.300E-04   AT NODE        105   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.009E-03   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     8.707E-06   AT NODE        636   DOF  6
   INSTANCE: INST                                                                            
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 47.0714E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 191.557E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.400       TIME AVG. FORCE       0.215    
 LARGEST RESIDUAL FORCE             9.543E-13   AT NODE        102   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.317E-07   AT NODE         29   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     6.563E-02   TIME AVG. MOMENT      3.530E-02
 LARGEST RESIDUAL MOMENT           -6.259E-15   AT NODE        103   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.007E-03   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     2.740E-06   AT NODE        453   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.260    
 STEP TIME COMPLETED       0.260    ,  TOTAL TIME COMPLETED        0.260    


  INCREMENT    27 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 40.4984E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 170.541E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.406       TIME AVG. FORCE       0.222    
 LARGEST RESIDUAL FORCE            -2.973E-03   AT NODE        310   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.207E-06   AT NODE        611   DOF  2
   INSTANCE: INST                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     6.747E-02   TIME AVG. MOMENT      3.649E-02
 LARGEST RESIDUAL MOMENT            1.234E-04   AT NODE        794   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -9.975E-04   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -1.017E-05   AT NODE        796   DOF  6
   INSTANCE: INST                                                                            
 ESTIMATE OF ROTATION CORRECTION   -1.508E-06
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 36.1876E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 231.679E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.411       TIME AVG. FORCE       0.222    
 LARGEST RESIDUAL FORCE            -9.097E-13   AT NODE        103   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -8.300E-08   AT NODE        837   DOF  1
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     6.747E-02   TIME AVG. MOMENT      3.649E-02
 LARGEST RESIDUAL MOMENT            5.634E-15   AT NODE        108   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -9.968E-04   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     6.924E-07   AT NODE        199   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.270    
 STEP TIME COMPLETED       0.270    ,  TOTAL TIME COMPLETED        0.270    


  INCREMENT    28 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 35.0610E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 277.581E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.422       TIME AVG. FORCE       0.230    
 LARGEST RESIDUAL FORCE            -1.075E-12   AT NODE        552   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -4.151E-07   AT NODE        795   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     6.923E-02   TIME AVG. MOMENT      3.766E-02
 LARGEST RESIDUAL MOMENT           -5.621E-15   AT NODE        121   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      9.984E-04   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     7.851E-06   AT NODE        310   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.280    
 STEP TIME COMPLETED       0.280    ,  TOTAL TIME COMPLETED        0.280    


  INCREMENT    29 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.414       TIME AVG. FORCE       0.236    
 LARGEST RESIDUAL FORCE            -1.418E-03   AT NODE        862   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.536E-05   AT NODE          5   DOF  2
   INSTANCE: INST                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     7.067E-02   TIME AVG. MOMENT      3.880E-02
 LARGEST RESIDUAL MOMENT            4.981E-05   AT NODE        494   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.071E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -1.492E-04   AT NODE        483   DOF  6
   INSTANCE: INST                                                                            
 ESTIMATE OF ROTATION CORRECTION    1.059E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 82.4111E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 609.793E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.431       TIME AVG. FORCE       0.236    
 LARGEST RESIDUAL FORCE             1.128E-12   AT NODE        113   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.170E-07   AT NODE        170   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     7.067E-02   TIME AVG. MOMENT      3.880E-02
 LARGEST RESIDUAL MOMENT           -5.967E-15   AT NODE         95   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.072E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     7.295E-07   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.290    
 STEP TIME COMPLETED       0.290    ,  TOTAL TIME COMPLETED        0.290    


  INCREMENT    30 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 63.5185E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 797.582E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.436       TIME AVG. FORCE       0.243    
 LARGEST RESIDUAL FORCE             1.263E-12   AT NODE        106   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.713E-05   AT NODE         20   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.150E-02   TIME AVG. MOMENT      3.989E-02
 LARGEST RESIDUAL MOMENT            6.467E-15   AT NODE        105   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.243E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     3.271E-04   AT NODE        175   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.300    
 STEP TIME COMPLETED       0.300    ,  TOTAL TIME COMPLETED        0.300    


  INCREMENT    31 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.433       TIME AVG. FORCE       0.249    
 LARGEST RESIDUAL FORCE            -1.233E-03   AT NODE        121   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.836E-05   AT NODE         18   DOF  2
   INSTANCE: INST                                                                            
 ESTIMATE OF DISP.    CORRECTION    8.849E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     7.203E-02   TIME AVG. MOMENT      4.092E-02
 LARGEST RESIDUAL MOMENT            5.119E-05   AT NODE         98   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.329E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.600E-04   AT NODE        836   DOF  6
   INSTANCE: INST                                                                            
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 63.5185E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 797.582E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.439       TIME AVG. FORCE       0.249    
 LARGEST RESIDUAL FORCE            -1.372E-12   AT NODE        562   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.876E-08   AT NODE          4   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     7.203E-02   TIME AVG. MOMENT      4.092E-02
 LARGEST RESIDUAL MOMENT            5.870E-15   AT NODE        116   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.330E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -2.373E-07   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.310    
 STEP TIME COMPLETED       0.310    ,  TOTAL TIME COMPLETED        0.310    


  INCREMENT    32 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 55.6569E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 685.320E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.442       TIME AVG. FORCE       0.255    
 LARGEST RESIDUAL FORCE            -1.413E-12   AT NODE        275   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.513E-07   AT NODE         11   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.256E-02   TIME AVG. MOMENT      4.191E-02
 LARGEST RESIDUAL MOMENT           -6.245E-15   AT NODE        570   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.328E-03   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.747E-06   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.320    
 STEP TIME COMPLETED       0.320    ,  TOTAL TIME COMPLETED        0.320    


  INCREMENT    33 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.446       TIME AVG. FORCE       0.261    
 LARGEST RESIDUAL FORCE            -1.272E-12   AT NODE        575   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.882E-07   AT NODE         20   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.309E-02   TIME AVG. MOMENT      4.286E-02
 LARGEST RESIDUAL MOMENT           -5.745E-15   AT NODE        113   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.326E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.976E-06   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.330    
 STEP TIME COMPLETED       0.330    ,  TOTAL TIME COMPLETED        0.330    


  INCREMENT    34 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.437       TIME AVG. FORCE       0.266    
 LARGEST RESIDUAL FORCE            -9.627E-03   AT NODE        174   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.062E-07   AT NODE        837   DOF  1
   INSTANCE: INST                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     7.362E-02   TIME AVG. MOMENT      4.376E-02
 LARGEST RESIDUAL MOMENT            1.079E-04   AT NODE        335   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.328E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     6.707E-06   AT NODE        335   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 43.1180E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 795.269E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.449       TIME AVG. FORCE       0.267    
 LARGEST RESIDUAL FORCE             1.319E-12   AT NODE        276   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.624E-06   AT NODE          1   DOF  1
   INSTANCE: {assembly node}                                                                 
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     7.360E-02   TIME AVG. MOMENT      4.376E-02
 LARGEST RESIDUAL MOMENT            6.578E-15   AT NODE        108   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.335E-03   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     4.043E-05   AT NODE        449   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.340    
 STEP TIME COMPLETED       0.340    ,  TOTAL TIME COMPLETED        0.340    


  INCREMENT    35 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 80.9435E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 2.80291E+15.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.452       TIME AVG. FORCE       0.272    
 LARGEST RESIDUAL FORCE            -1.370E-12   AT NODE        125   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.182E-06   AT NODE        819   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.410E-02   TIME AVG. MOMENT      4.463E-02
 LARGEST RESIDUAL MOMENT           -6.689E-15   AT NODE         97   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.339E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -1.936E-05   AT NODE        771   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.350    
 STEP TIME COMPLETED       0.350    ,  TOTAL TIME COMPLETED        0.350    


  INCREMENT    36 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.449       TIME AVG. FORCE       0.277    
 LARGEST RESIDUAL FORCE            -1.289E-03   AT NODE        799   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.385E-07   AT NODE        449   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     7.460E-02   TIME AVG. MOMENT      4.546E-02
 LARGEST RESIDUAL MOMENT           -5.351E-05   AT NODE        196   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.337E-03   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.983E-06   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.360    
 STEP TIME COMPLETED       0.360    ,  TOTAL TIME COMPLETED        0.360    


  INCREMENT    37 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 96.0873E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 300.954E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.458       TIME AVG. FORCE       0.282    
 LARGEST RESIDUAL FORCE             1.809E-12   AT NODE        111   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.785E-07   AT NODE        335   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.510E-02   TIME AVG. MOMENT      4.626E-02
 LARGEST RESIDUAL MOMENT            7.633E-15   AT NODE        110   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.335E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.693E-06   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.370    
 STEP TIME COMPLETED       0.370    ,  TOTAL TIME COMPLETED        0.370    


  INCREMENT    38 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.455       TIME AVG. FORCE       0.286    
 LARGEST RESIDUAL FORCE            -6.969E-03   AT NODE        335   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.990E-07   AT NODE        610   DOF  2
   INSTANCE: INST                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     7.560E-02   TIME AVG. MOMENT      4.703E-02
 LARGEST RESIDUAL MOMENT           -7.811E-05   AT NODE        450   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.337E-03   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     5.881E-06   AT NODE        335   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 86.3096E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 454.158E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.461       TIME AVG. FORCE       0.287    
 LARGEST RESIDUAL FORCE            -1.311E-12   AT NODE        115   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.683E-06   AT NODE        610   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     7.555E-02   TIME AVG. MOMENT      4.703E-02
 LARGEST RESIDUAL MOMENT            6.287E-15   AT NODE        389   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.359E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -1.030E-04   AT NODE        611   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.380    
 STEP TIME COMPLETED       0.380    ,  TOTAL TIME COMPLETED        0.380    


  INCREMENT    39 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 56.7141E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 622.243E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.463       TIME AVG. FORCE       0.291    
 LARGEST RESIDUAL FORCE             1.376E-12   AT NODE         89   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.072E-05   AT NODE        885   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.588E-02   TIME AVG. MOMENT      4.777E-02
 LARGEST RESIDUAL MOMENT            6.738E-15   AT NODE        140   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.418E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -2.734E-04   AT NODE        836   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.390    
 STEP TIME COMPLETED       0.390    ,  TOTAL TIME COMPLETED        0.390    


  INCREMENT    40 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.459       TIME AVG. FORCE       0.295    
 LARGEST RESIDUAL FORCE            -4.109E-04   AT NODE        193   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -8.345E-08   AT NODE          1   DOF  1
   INSTANCE: {assembly node}                                                                 
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     7.621E-02   TIME AVG. MOMENT      4.848E-02
 LARGEST RESIDUAL MOMENT           -1.706E-05   AT NODE        801   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.419E-03   AT NODE         26   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.096E-06   AT NODE        890   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.400    
 STEP TIME COMPLETED       0.400    ,  TOTAL TIME COMPLETED        0.400    


  INCREMENT    41 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 52.5806E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 7.31689E+15.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.467       TIME AVG. FORCE       0.299    
 LARGEST RESIDUAL FORCE            -1.501E-12   AT NODE        555   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.181E-06   AT NODE        175   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.654E-02   TIME AVG. MOMENT      4.917E-02
 LARGEST RESIDUAL MOMENT            8.868E-15   AT NODE        123   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.417E-03   AT NODE         27   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.752E-06   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.410    
 STEP TIME COMPLETED       0.410    ,  TOTAL TIME COMPLETED        0.410    


  INCREMENT    42 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.469       TIME AVG. FORCE       0.303    
 LARGEST RESIDUAL FORCE            -1.435E-12   AT NODE        378   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.138E-07   AT NODE        819   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.687E-02   TIME AVG. MOMENT      4.983E-02
 LARGEST RESIDUAL MOMENT            8.563E-15   AT NODE        377   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.417E-03   AT NODE         26   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     3.259E-07   AT NODE        527   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.420    
 STEP TIME COMPLETED       0.420    ,  TOTAL TIME COMPLETED        0.420    


  INCREMENT    43 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.471       TIME AVG. FORCE       0.307    
 LARGEST RESIDUAL FORCE            -1.647E-12   AT NODE        122   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.548E-15   AT NODE        376   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.719E-02   TIME AVG. MOMENT      5.046E-02
 LARGEST RESIDUAL MOMENT            7.688E-15   AT NODE        557   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.417E-03   AT NODE         26   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     3.879E-16   AT NODE        350   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.430    
 STEP TIME COMPLETED       0.430    ,  TOTAL TIME COMPLETED        0.430    


  INCREMENT    44 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.473       TIME AVG. FORCE       0.311    
 LARGEST RESIDUAL FORCE            -1.592E-12   AT NODE         89   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.110E-16   AT NODE        143   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.752E-02   TIME AVG. MOMENT      5.108E-02
 LARGEST RESIDUAL MOMENT           -7.314E-15   AT NODE        376   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.417E-03   AT NODE         27   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.967E-16   AT NODE        154   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.440    
 STEP TIME COMPLETED       0.440    ,  TOTAL TIME COMPLETED        0.440    


  INCREMENT    45 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.468       TIME AVG. FORCE       0.315    
 LARGEST RESIDUAL FORCE            -1.558E-04   AT NODE        837   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -3.142E-08   AT NODE        772   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     7.785E-02   TIME AVG. MOMENT      5.167E-02
 LARGEST RESIDUAL MOMENT            1.746E-06   AT NODE        820   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.417E-03   AT NODE         26   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -9.469E-08   AT NODE        174   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.450    
 STEP TIME COMPLETED       0.450    ,  TOTAL TIME COMPLETED        0.450    


  INCREMENT    46 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 69.9338E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 1.90164E+15.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.477       TIME AVG. FORCE       0.318    
 LARGEST RESIDUAL FORCE            -1.637E-12   AT NODE        120   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.484E-06   AT NODE        885   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.817E-02   TIME AVG. MOMENT      5.225E-02
 LARGEST RESIDUAL MOMENT            6.786E-15   AT NODE         87   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.423E-03   AT NODE         27   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     4.130E-05   AT NODE        336   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.460    
 STEP TIME COMPLETED       0.460    ,  TOTAL TIME COMPLETED        0.460    


  INCREMENT    47 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.466       TIME AVG. FORCE       0.321    
 LARGEST RESIDUAL FORCE            -1.170E-03   AT NODE        801   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.241E-07   AT NODE        885   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     7.848E-02   TIME AVG. MOMENT      5.281E-02
 LARGEST RESIDUAL MOMENT            3.659E-05   AT NODE        124   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.423E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     7.963E-07   AT NODE         27   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.470    
 STEP TIME COMPLETED       0.470    ,  TOTAL TIME COMPLETED        0.470    


  INCREMENT    48 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 39.7526E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 647.200E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.480       TIME AVG. FORCE       0.325    
 LARGEST RESIDUAL FORCE            -2.303E-12   AT NODE         98   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.727E-07   AT NODE        335   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.880E-02   TIME AVG. MOMENT      5.335E-02
 LARGEST RESIDUAL MOMENT            8.438E-15   AT NODE        546   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.421E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.670E-06   AT NODE         30   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.480    
 STEP TIME COMPLETED       0.480    ,  TOTAL TIME COMPLETED        0.480    


  INCREMENT    49 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.482       TIME AVG. FORCE       0.328    
 LARGEST RESIDUAL FORCE             2.068E-12   AT NODE         98   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        5.638E-08   AT NODE        141   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.912E-02   TIME AVG. MOMENT      5.388E-02
 LARGEST RESIDUAL MOMENT            1.293E-14   AT NODE          1   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.421E-03   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     3.532E-07   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.490    
 STEP TIME COMPLETED       0.490    ,  TOTAL TIME COMPLETED        0.490    


  INCREMENT    50 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.484       TIME AVG. FORCE       0.331    
 LARGEST RESIDUAL FORCE            -1.694E-12   AT NODE        286   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        8.401E-17   AT NODE        899   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.943E-02   TIME AVG. MOMENT      5.439E-02
 LARGEST RESIDUAL MOMENT            9.895E-15   AT NODE        124   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.421E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -2.433E-16   AT NODE        589   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.500    
 STEP TIME COMPLETED       0.500    ,  TOTAL TIME COMPLETED        0.500    


  INCREMENT    51 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.486       TIME AVG. FORCE       0.334    
 LARGEST RESIDUAL FORCE             1.594E-12   AT NODE        127   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -5.765E-17   AT NODE        899   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.975E-02   TIME AVG. MOMENT      5.488E-02
 LARGEST RESIDUAL MOMENT            8.590E-15   AT NODE        121   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.421E-03   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.512E-16   AT NODE        601   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.510    
 STEP TIME COMPLETED       0.510    ,  TOTAL TIME COMPLETED        0.510    


  INCREMENT    52 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.488       TIME AVG. FORCE       0.337    
 LARGEST RESIDUAL FORCE             1.962E-12   AT NODE        215   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        9.038E-17   AT NODE        820   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.007E-02   TIME AVG. MOMENT      5.537E-02
 LARGEST RESIDUAL MOMENT           -1.268E-14   AT NODE         93   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.421E-03   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.683E-16   AT NODE        829   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.520    
 STEP TIME COMPLETED       0.520    ,  TOTAL TIME COMPLETED        0.520    


  INCREMENT    53 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.477       TIME AVG. FORCE       0.340    
 LARGEST RESIDUAL FORCE            -1.897E-03   AT NODE        609   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        7.143E-08   AT NODE        609   DOF  2
   INSTANCE: INST                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     8.038E-02   TIME AVG. MOMENT      5.584E-02
 LARGEST RESIDUAL MOMENT            2.126E-05   AT NODE        336   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.422E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     2.026E-06   AT NODE        450   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 38.5774E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 334.564E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.490       TIME AVG. FORCE       0.340    
 LARGEST RESIDUAL FORCE            -2.236E-12   AT NODE         98   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.029E-06   AT NODE        451   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     8.037E-02   TIME AVG. MOMENT      5.584E-02
 LARGEST RESIDUAL MOMENT            1.138E-14   AT NODE        125   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.429E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -3.596E-05   AT NODE        610   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.530    
 STEP TIME COMPLETED       0.530    ,  TOTAL TIME COMPLETED        0.530    


  INCREMENT    54 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 45.9846E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 436.696E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.485       TIME AVG. FORCE       0.343    
 LARGEST RESIDUAL FORCE             4.195E-04   AT NODE        318   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -7.416E-06   AT NODE        821   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     8.062E-02   TIME AVG. MOMENT      5.630E-02
 LARGEST RESIDUAL MOMENT            1.742E-05   AT NODE        628   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.480E-03   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -2.548E-04   AT NODE        772   DOF  6
   INSTANCE: INST                                                                            
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 52.3882E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 188.550E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.492       TIME AVG. FORCE       0.343    
 LARGEST RESIDUAL FORCE            -2.498E-12   AT NODE         84   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.403E-08   AT NODE        449   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     8.062E-02   TIME AVG. MOMENT      5.630E-02
 LARGEST RESIDUAL MOMENT            1.030E-14   AT NODE        116   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.480E-03   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -9.987E-08   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.540    
 STEP TIME COMPLETED       0.540    ,  TOTAL TIME COMPLETED        0.540    


  INCREMENT    55 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 45.9846E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 436.696E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.493       TIME AVG. FORCE       0.345    
 LARGEST RESIDUAL FORCE            -3.163E-12   AT NODE         84   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.332E-07   AT NODE        577   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.086E-02   TIME AVG. MOMENT      5.675E-02
 LARGEST RESIDUAL MOMENT            1.571E-14   AT NODE         85   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.479E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -8.970E-07   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.550    
 STEP TIME COMPLETED       0.550    ,  TOTAL TIME COMPLETED        0.550    


  INCREMENT    56 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.495       TIME AVG. FORCE       0.348    
 LARGEST RESIDUAL FORCE             2.191E-12   AT NODE        114   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.483E-08   AT NODE        175   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.111E-02   TIME AVG. MOMENT      5.718E-02
 LARGEST RESIDUAL MOMENT            1.010E-14   AT NODE         96   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.479E-03   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     9.987E-08   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.560    
 STEP TIME COMPLETED       0.560    ,  TOTAL TIME COMPLETED        0.560    


  INCREMENT    57 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.496       TIME AVG. FORCE       0.351    
 LARGEST RESIDUAL FORCE            -2.195E-12   AT NODE         92   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -4.447E-12   AT NODE          6   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.135E-02   TIME AVG. MOMENT      5.760E-02
 LARGEST RESIDUAL MOMENT            1.094E-14   AT NODE         13   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.479E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     2.460E-10   AT NODE        611   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.570    
 STEP TIME COMPLETED       0.570    ,  TOTAL TIME COMPLETED        0.570    


  INCREMENT    58 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.497       TIME AVG. FORCE       0.353    
 LARGEST RESIDUAL FORCE             3.356E-12   AT NODE         95   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.180E-11   AT NODE         16   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.160E-02   TIME AVG. MOMENT      5.802E-02
 LARGEST RESIDUAL MOMENT            1.621E-14   AT NODE         94   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.479E-03   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -6.309E-10   AT NODE        483   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.580    
 STEP TIME COMPLETED       0.580    ,  TOTAL TIME COMPLETED        0.580    


  INCREMENT    59 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.499       TIME AVG. FORCE       0.356    
 LARGEST RESIDUAL FORCE             2.200E-12   AT NODE        396   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -4.084E-12   AT NODE         25   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.184E-02   TIME AVG. MOMENT      5.842E-02
 LARGEST RESIDUAL MOMENT           -1.379E-14   AT NODE        124   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.479E-03   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     2.117E-10   AT NODE        836   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.590    
 STEP TIME COMPLETED       0.590    ,  TOTAL TIME COMPLETED        0.590    


  INCREMENT    60 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.500       TIME AVG. FORCE       0.358    
 LARGEST RESIDUAL FORCE            -2.564E-12   AT NODE         91   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.088E-16   AT NODE         25   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.209E-02   TIME AVG. MOMENT      5.882E-02
 LARGEST RESIDUAL MOMENT            1.299E-14   AT NODE         92   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.479E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -6.562E-16   AT NODE        141   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.600    
 STEP TIME COMPLETED       0.600    ,  TOTAL TIME COMPLETED        0.600    


  INCREMENT    61 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.502       TIME AVG. FORCE       0.360    
 LARGEST RESIDUAL FORCE            -2.963E-12   AT NODE        551   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        5.419E-13   AT NODE        175   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.233E-02   TIME AVG. MOMENT      5.920E-02
 LARGEST RESIDUAL MOMENT           -1.232E-14   AT NODE        549   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.479E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.641E-11   AT NODE        336   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.610    
 STEP TIME COMPLETED       0.610    ,  TOTAL TIME COMPLETED        0.610    


  INCREMENT    62 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.503       TIME AVG. FORCE       0.363    
 LARGEST RESIDUAL FORCE             2.191E-12   AT NODE        112   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.077E-11   AT NODE        819   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.258E-02   TIME AVG. MOMENT      5.958E-02
 LARGEST RESIDUAL MOMENT            1.324E-14   AT NODE        111   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.479E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -3.243E-10   AT NODE        772   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.620    
 STEP TIME COMPLETED       0.620    ,  TOTAL TIME COMPLETED        0.620    


  INCREMENT    63 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.498       TIME AVG. FORCE       0.365    
 LARGEST RESIDUAL FORCE             2.128E-04   AT NODE        868   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.797E-08   AT NODE        771   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     8.282E-02   TIME AVG. MOMENT      5.995E-02
 LARGEST RESIDUAL MOMENT            8.833E-06   AT NODE        531   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.479E-03   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -1.170E-07   AT NODE         26   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.630    
 STEP TIME COMPLETED       0.630    ,  TOTAL TIME COMPLETED        0.630    


  INCREMENT    64 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 37.6758E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 511.420E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.506       TIME AVG. FORCE       0.367    
 LARGEST RESIDUAL FORCE            -2.156E-12   AT NODE        230   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.726E-07   AT NODE          1   DOF  1
   INSTANCE: {assembly node}                                                                 
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.307E-02   TIME AVG. MOMENT      6.031E-02
 LARGEST RESIDUAL MOMENT           -1.054E-14   AT NODE        229   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.478E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -2.844E-06   AT NODE        609   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.640    
 STEP TIME COMPLETED       0.640    ,  TOTAL TIME COMPLETED        0.640    


  INCREMENT    65 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.508       TIME AVG. FORCE       0.369    
 LARGEST RESIDUAL FORCE             5.014E-10   AT NODE        368   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.672E-06   AT NODE          1   DOF  1
   INSTANCE: {assembly node}                                                                 
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.331E-02   TIME AVG. MOMENT      6.066E-02
 LARGEST RESIDUAL MOMENT            5.619E-12   AT NODE        771   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.483E-03   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     2.705E-05   AT NODE        821   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.650    
 STEP TIME COMPLETED       0.650    ,  TOTAL TIME COMPLETED        0.650    


  INCREMENT    66 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.509       TIME AVG. FORCE       0.371    
 LARGEST RESIDUAL FORCE            -2.250E-12   AT NODE        116   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -5.052E-12   AT NODE        578   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.355E-02   TIME AVG. MOMENT      6.101E-02
 LARGEST RESIDUAL MOMENT           -1.389E-14   AT NODE        380   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.483E-03   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.225E-10   AT NODE        449   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.660    
 STEP TIME COMPLETED       0.660    ,  TOTAL TIME COMPLETED        0.660    


  INCREMENT    67 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.504       TIME AVG. FORCE       0.373    
 LARGEST RESIDUAL FORCE            -3.301E-04   AT NODE        585   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.054E-08   AT NODE        476   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     8.378E-02   TIME AVG. MOMENT      6.135E-02
 LARGEST RESIDUAL MOMENT           -7.400E-06   AT NODE        361   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.483E-03   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -3.668E-07   AT NODE        584   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.670    
 STEP TIME COMPLETED       0.670    ,  TOTAL TIME COMPLETED        0.670    


  INCREMENT    68 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 31.2787E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 3.51116E+15.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.512       TIME AVG. FORCE       0.375    
 LARGEST RESIDUAL FORCE            -3.330E-12   AT NODE        101   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        7.992E-08   AT NODE        891   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.402E-02   TIME AVG. MOMENT      6.168E-02
 LARGEST RESIDUAL MOMENT           -1.567E-14   AT NODE        565   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.483E-03   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -3.781E-07   AT NODE        604   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.680    
 STEP TIME COMPLETED       0.680    ,  TOTAL TIME COMPLETED        0.680    


  INCREMENT    69 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.514       TIME AVG. FORCE       0.377    
 LARGEST RESIDUAL FORCE             2.433E-12   AT NODE         97   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -3.909E-08   AT NODE        476   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.426E-02   TIME AVG. MOMENT      6.201E-02
 LARGEST RESIDUAL MOMENT           -1.550E-14   AT NODE        564   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.483E-03   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.804E-07   AT NODE        584   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.690    
 STEP TIME COMPLETED       0.690    ,  TOTAL TIME COMPLETED        0.690    


  INCREMENT    70 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.515       TIME AVG. FORCE       0.379    
 LARGEST RESIDUAL FORCE            -2.430E-12   AT NODE        134   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -8.477E-16   AT NODE        174   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.450E-02   TIME AVG. MOMENT      6.233E-02
 LARGEST RESIDUAL MOMENT            1.382E-14   AT NODE        101   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.483E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     4.017E-16   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.700    
 STEP TIME COMPLETED       0.700    ,  TOTAL TIME COMPLETED        0.700    


  INCREMENT    71 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.510       TIME AVG. FORCE       0.381    
 LARGEST RESIDUAL FORCE            -6.942E-03   AT NODE          7   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.196E-06   AT NODE        771   DOF  2
   INSTANCE: INST                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     8.474E-02   TIME AVG. MOMENT      6.265E-02
 LARGEST RESIDUAL MOMENT           -1.248E-04   AT NODE        484   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.481E-03   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -2.423E-06   AT NODE         28   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 28.6309E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 6.08358E+15.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.517       TIME AVG. FORCE       0.381    
 LARGEST RESIDUAL FORCE             2.951E-12   AT NODE         88   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        6.688E-06   AT NODE        771   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     8.474E-02   TIME AVG. MOMENT      6.265E-02
 LARGEST RESIDUAL MOMENT           -1.414E-14   AT NODE        138   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.463E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -1.787E-05   AT NODE         28   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.710    
 STEP TIME COMPLETED       0.710    ,  TOTAL TIME COMPLETED        0.710    


  INCREMENT    72 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 25.3876E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 1.04237E+15.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.518       TIME AVG. FORCE       0.383    
 LARGEST RESIDUAL FORCE            -2.444E-12   AT NODE        393   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.157E-06   AT NODE        577   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.497E-02   TIME AVG. MOMENT      6.296E-02
 LARGEST RESIDUAL MOMENT           -1.521E-14   AT NODE        135   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.450E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -1.317E-05   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.720    
 STEP TIME COMPLETED       0.720    ,  TOTAL TIME COMPLETED        0.720    


  INCREMENT    73 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.519       TIME AVG. FORCE       0.385    
 LARGEST RESIDUAL FORCE             3.106E-12   AT NODE         95   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.016E-11   AT NODE         19   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.520E-02   TIME AVG. MOMENT      6.326E-02
 LARGEST RESIDUAL MOMENT            1.406E-14   AT NODE         94   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.450E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     5.563E-10   AT NODE        369   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.730    
 STEP TIME COMPLETED       0.730    ,  TOTAL TIME COMPLETED        0.730    


  INCREMENT    74 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.514       TIME AVG. FORCE       0.387    
 LARGEST RESIDUAL FORCE             5.038E-04   AT NODE        756   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        6.160E-08   AT NODE        820   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     8.543E-02   TIME AVG. MOMENT      6.356E-02
 LARGEST RESIDUAL MOMENT           -2.092E-05   AT NODE        501   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.449E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -2.639E-07   AT NODE        501   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.740    
 STEP TIME COMPLETED       0.740    ,  TOTAL TIME COMPLETED        0.740    


  INCREMENT    75 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 83.5604E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 20.3913E+15 .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.522       TIME AVG. FORCE       0.389    
 LARGEST RESIDUAL FORCE            -2.826E-12   AT NODE        139   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.960E-07   AT NODE        772   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.566E-02   TIME AVG. MOMENT      6.386E-02
 LARGEST RESIDUAL MOMENT            1.625E-14   AT NODE        574   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.449E-03   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     7.999E-07   AT NODE        757   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.750    
 STEP TIME COMPLETED       0.750    ,  TOTAL TIME COMPLETED        0.750    


  INCREMENT    76 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.524       TIME AVG. FORCE       0.390    
 LARGEST RESIDUAL FORCE            -6.914E-12   AT NODE        141   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.572E-07   AT NODE        368   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.589E-02   TIME AVG. MOMENT      6.415E-02
 LARGEST RESIDUAL MOMENT            7.291E-14   AT NODE        577   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.448E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -3.124E-07   AT NODE        384   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.760    
 STEP TIME COMPLETED       0.760    ,  TOTAL TIME COMPLETED        0.760    


  INCREMENT    77 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.518       TIME AVG. FORCE       0.392    
 LARGEST RESIDUAL FORCE             9.691E-03   AT NODE        338   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.064E-06   AT NODE        608   DOF  2
   INSTANCE: INST                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     8.612E-02   TIME AVG. MOMENT      6.443E-02
 LARGEST RESIDUAL MOMENT           -1.086E-04   AT NODE        338   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.451E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -9.131E-06   AT NODE        609   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 63.9372E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 6.16864E+15.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.525       TIME AVG. FORCE       0.392    
 LARGEST RESIDUAL FORCE             2.880E-12   AT NODE        128   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.071E-05   AT NODE        452   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     8.609E-02   TIME AVG. MOMENT      6.443E-02
 LARGEST RESIDUAL MOMENT            1.119E-14   AT NODE        101   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.501E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -1.966E-04   AT NODE        173   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.770    
 STEP TIME COMPLETED       0.770    ,  TOTAL TIME COMPLETED        0.770    


  INCREMENT    78 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 70.9501E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 476.917E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.526       TIME AVG. FORCE       0.394    
 LARGEST RESIDUAL FORCE            -3.131E-12   AT NODE        129   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.057E-05   AT NODE        771   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.626E-02   TIME AVG. MOMENT      6.471E-02
 LARGEST RESIDUAL MOMENT            1.722E-14   AT NODE        130   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.378E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -1.231E-04   AT NODE         26   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.780    
 STEP TIME COMPLETED       0.780    ,  TOTAL TIME COMPLETED        0.780    


  INCREMENT    79 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.520       TIME AVG. FORCE       0.395    
 LARGEST RESIDUAL FORCE            -3.847E-03   AT NODE        484   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.561E-05   AT NODE        577   DOF  2
   INSTANCE: INST                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     8.641E-02   TIME AVG. MOMENT      6.499E-02
 LARGEST RESIDUAL MOMENT           -7.986E-05   AT NODE        740   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.284E-03   AT NODE         26   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -9.426E-05   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 62.6525E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 400.064E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.527       TIME AVG. FORCE       0.396    
 LARGEST RESIDUAL FORCE            -2.925E-12   AT NODE        133   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.790E-06   AT NODE        771   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     8.641E-02   TIME AVG. MOMENT      6.499E-02
 LARGEST RESIDUAL MOMENT            1.429E-14   AT NODE         85   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.275E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     8.913E-06   AT NODE         27   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.790    
 STEP TIME COMPLETED       0.790    ,  TOTAL TIME COMPLETED        0.790    


  INCREMENT    80 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 62.6525E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 400.064E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.528       TIME AVG. FORCE       0.397    
 LARGEST RESIDUAL FORCE             3.351E-12   AT NODE         87   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.163E-06   AT NODE        449   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.655E-02   TIME AVG. MOMENT      6.526E-02
 LARGEST RESIDUAL MOMENT           -1.887E-14   AT NODE        129   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.269E-03   AT NODE         26   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     5.792E-06   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.800    
 STEP TIME COMPLETED       0.800    ,  TOTAL TIME COMPLETED        0.800    


  INCREMENT    81 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.529       TIME AVG. FORCE       0.399    
 LARGEST RESIDUAL FORCE            -3.235E-12   AT NODE        385   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.332E-16   AT NODE         26   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.670E-02   TIME AVG. MOMENT      6.552E-02
 LARGEST RESIDUAL MOMENT            1.354E-14   AT NODE         13   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.269E-03   AT NODE         26   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     4.354E-16   AT NODE        470   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.810    
 STEP TIME COMPLETED       0.810    ,  TOTAL TIME COMPLETED        0.810    


  INCREMENT    82 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.530       TIME AVG. FORCE       0.400    
 LARGEST RESIDUAL FORCE             3.493E-12   AT NODE        113   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -5.554E-16   AT NODE         26   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.685E-02   TIME AVG. MOMENT      6.578E-02
 LARGEST RESIDUAL MOMENT           -1.329E-14   AT NODE         95   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.269E-03   AT NODE         26   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -9.416E-16   AT NODE        472   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.820    
 STEP TIME COMPLETED       0.820    ,  TOTAL TIME COMPLETED        0.820    


  INCREMENT    83 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.530       TIME AVG. FORCE       0.402    
 LARGEST RESIDUAL FORCE             3.194E-12   AT NODE        397   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        9.962E-17   AT NODE         12   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.700E-02   TIME AVG. MOMENT      6.604E-02
 LARGEST RESIDUAL MOMENT            1.613E-14   AT NODE        107   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.269E-03   AT NODE         27   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     3.183E-16   AT NODE        340   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.830    
 STEP TIME COMPLETED       0.830    ,  TOTAL TIME COMPLETED        0.830    


  INCREMENT    84 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.531       TIME AVG. FORCE       0.404    
 LARGEST RESIDUAL FORCE             2.469E-12   AT NODE        576   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -6.175E-17   AT NODE        835   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.715E-02   TIME AVG. MOMENT      6.629E-02
 LARGEST RESIDUAL MOMENT           -1.166E-14   AT NODE        117   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.269E-03   AT NODE         26   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     2.230E-16   AT NODE        847   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.840    
 STEP TIME COMPLETED       0.840    ,  TOTAL TIME COMPLETED        0.840    


  INCREMENT    85 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.532       TIME AVG. FORCE       0.405    
 LARGEST RESIDUAL FORCE            -4.110E-12   AT NODE        115   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.055E-16   AT NODE         17   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.730E-02   TIME AVG. MOMENT      6.653E-02
 LARGEST RESIDUAL MOMENT            1.435E-14   AT NODE        372   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.269E-03   AT NODE         26   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     2.614E-16   AT NODE        837   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.850    
 STEP TIME COMPLETED       0.850    ,  TOTAL TIME COMPLETED        0.850    


  INCREMENT    86 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.533       TIME AVG. FORCE       0.407    
 LARGEST RESIDUAL FORCE             3.351E-12   AT NODE        122   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.267E-16   AT NODE        462   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.744E-02   TIME AVG. MOMENT      6.678E-02
 LARGEST RESIDUAL MOMENT           -1.617E-14   AT NODE        123   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.269E-03   AT NODE         27   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -6.601E-16   AT NODE        784   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.860    
 STEP TIME COMPLETED       0.860    ,  TOTAL TIME COMPLETED        0.860    


  INCREMENT    87 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.534       TIME AVG. FORCE       0.408    
 LARGEST RESIDUAL FORCE            -2.668E-12   AT NODE        555   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.265E-16   AT NODE        772   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.759E-02   TIME AVG. MOMENT      6.702E-02
 LARGEST RESIDUAL MOMENT           -1.370E-14   AT NODE        568   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.269E-03   AT NODE         26   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     5.903E-16   AT NODE        779   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.870    
 STEP TIME COMPLETED       0.870    ,  TOTAL TIME COMPLETED        0.870    


  INCREMENT    88 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.535       TIME AVG. FORCE       0.409    
 LARGEST RESIDUAL FORCE             3.245E-12   AT NODE        570   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.685E-11   AT NODE        173   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.774E-02   TIME AVG. MOMENT      6.725E-02
 LARGEST RESIDUAL MOMENT            1.868E-14   AT NODE        371   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.269E-03   AT NODE         27   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.646E-10   AT NODE        610   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.880    
 STEP TIME COMPLETED       0.880    ,  TOTAL TIME COMPLETED        0.880    


  INCREMENT    89 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.536       TIME AVG. FORCE       0.411    
 LARGEST RESIDUAL FORCE            -2.860E-12   AT NODE        383   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        6.586E-12   AT NODE        451   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.789E-02   TIME AVG. MOMENT      6.748E-02
 LARGEST RESIDUAL MOMENT            1.948E-14   AT NODE        101   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.269E-03   AT NODE         26   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -6.381E-11   AT NODE        450   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.890    
 STEP TIME COMPLETED       0.890    ,  TOTAL TIME COMPLETED        0.890    


  INCREMENT    90 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.523       TIME AVG. FORCE       0.412    
 LARGEST RESIDUAL FORCE             6.704E-03   AT NODE        515   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        6.703E-07   AT NODE        771   DOF  2
   INSTANCE: INST                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     8.803E-02   TIME AVG. MOMENT      6.771E-02
 LARGEST RESIDUAL MOMENT            1.392E-04   AT NODE        484   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.265E-03   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -3.436E-06   AT NODE         28   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 70.8909E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 349.182E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.537       TIME AVG. FORCE       0.412    
 LARGEST RESIDUAL FORCE             4.170E-12   AT NODE        546   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.102E-05   AT NODE        771   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     8.802E-02   TIME AVG. MOMENT      6.771E-02
 LARGEST RESIDUAL MOMENT            1.835E-14   AT NODE        567   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.196E-03   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -6.970E-05   AT NODE         26   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.900    
 STEP TIME COMPLETED       0.900    ,  TOTAL TIME COMPLETED        0.900    


  INCREMENT    91 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 75.8627E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 475.251E+12  .

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.537       TIME AVG. FORCE       0.414    
 LARGEST RESIDUAL FORCE             3.960E-12   AT NODE        105   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.003E-05   AT NODE        449   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.814E-02   TIME AVG. MOMENT      6.794E-02
 LARGEST RESIDUAL MOMENT           -1.715E-14   AT NODE        106   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.138E-03   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     5.755E-05   AT NODE        466   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.910    
 STEP TIME COMPLETED       0.910    ,  TOTAL TIME COMPLETED        0.910    


  INCREMENT    92 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.531       TIME AVG. FORCE       0.415    
 LARGEST RESIDUAL FORCE            -2.617E-04   AT NODE        869   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.082E-06   AT NODE        175   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     8.826E-02   TIME AVG. MOMENT      6.816E-02
 LARGEST RESIDUAL MOMENT           -1.086E-05   AT NODE        756   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.132E-03   AT NODE         26   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -6.351E-06   AT NODE        158   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.920    
 STEP TIME COMPLETED       0.920    ,  TOTAL TIME COMPLETED        0.920    


  INCREMENT    93 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 80.8330E+12 .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 2.47465E+15.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.539       TIME AVG. FORCE       0.416    
 LARGEST RESIDUAL FORCE             2.494E-12   AT NODE        400   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.612E-08   AT NODE        610   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.838E-02   TIME AVG. MOMENT      6.838E-02
 LARGEST RESIDUAL MOMENT           -1.846E-14   AT NODE         97   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.132E-03   AT NODE         26   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     2.466E-07   AT NODE        625   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.930    
 STEP TIME COMPLETED       0.930    ,  TOTAL TIME COMPLETED        0.930    


  INCREMENT    94 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.540       TIME AVG. FORCE       0.418    
 LARGEST RESIDUAL FORCE             3.577E-12   AT NODE         93   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        7.812E-09   AT NODE        142   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.850E-02   TIME AVG. MOMENT      6.859E-02
 LARGEST RESIDUAL MOMENT           -3.006E-14   AT NODE          5   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.132E-03   AT NODE         26   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -6.930E-08   AT NODE        563   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.940    
 STEP TIME COMPLETED       0.940    ,  TOTAL TIME COMPLETED        0.940    


  INCREMENT    95 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.540       TIME AVG. FORCE       0.419    
 LARGEST RESIDUAL FORCE             4.230E-12   AT NODE        105   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -5.724E-11   AT NODE          6   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.862E-02   TIME AVG. MOMENT      6.880E-02
 LARGEST RESIDUAL MOMENT           -1.693E-14   AT NODE          1   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.132E-03   AT NODE         27   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     2.398E-10   AT NODE        579   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.950    
 STEP TIME COMPLETED       0.950    ,  TOTAL TIME COMPLETED        0.950    


  INCREMENT    96 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.541       TIME AVG. FORCE       0.420    
 LARGEST RESIDUAL FORCE             3.754E-12   AT NODE        556   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.737E-11   AT NODE        819   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.874E-02   TIME AVG. MOMENT      6.901E-02
 LARGEST RESIDUAL MOMENT            2.137E-14   AT NODE        116   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.132E-03   AT NODE         26   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.176E-10   AT NODE        773   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.960    
 STEP TIME COMPLETED       0.960    ,  TOTAL TIME COMPLETED        0.960    


  INCREMENT    97 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.542       TIME AVG. FORCE       0.421    
 LARGEST RESIDUAL FORCE             3.612E-12   AT NODE         90   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -3.122E-11   AT NODE         25   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.886E-02   TIME AVG. MOMENT      6.921E-02
 LARGEST RESIDUAL MOMENT            2.398E-14   AT NODE        105   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.132E-03   AT NODE         27   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     2.067E-10   AT NODE        836   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.970    
 STEP TIME COMPLETED       0.970    ,  TOTAL TIME COMPLETED        0.970    


  INCREMENT    98 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.535       TIME AVG. FORCE       0.423    
 LARGEST RESIDUAL FORCE            -5.409E-05   AT NODE        361   DOF  1
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.202E-08   AT NODE        585   DOF  2
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     8.898E-02   TIME AVG. MOMENT      6.941E-02
 LARGEST RESIDUAL MOMENT            1.213E-06   AT NODE        476   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION     -1.132E-03   AT NODE        352   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -7.144E-08   AT NODE        361   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.980    
 STEP TIME COMPLETED       0.980    ,  TOTAL TIME COMPLETED        0.980    


  INCREMENT    99 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 1 RATIO = 1.67212E+15.
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE INST.15 
             D.O.F. 2 RATIO = 2.11550E+15.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.543       TIME AVG. FORCE       0.424    
 LARGEST RESIDUAL FORCE            -3.318E-12   AT NODE        104   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.761E-06   AT NODE        823   DOF  1
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.910E-02   TIME AVG. MOMENT      6.961E-02
 LARGEST RESIDUAL MOMENT           -1.954E-14   AT NODE        112   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.138E-03   AT NODE        594   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION    -8.694E-06   AT NODE        144   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.990    
 STEP TIME COMPLETED       0.990    ,  TOTAL TIME COMPLETED        0.990    


  INCREMENT   100 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02

  NUMBER OF EQUATIONS =      2693     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.90E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.544       TIME AVG. FORCE       0.425    
 LARGEST RESIDUAL FORCE             4.324E-12   AT NODE        103   DOF  2
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF DISP.        -9.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        4.112E-07   AT NODE        840   DOF  1
   INSTANCE: INST                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.922E-02   TIME AVG. MOMENT      6.981E-02
 LARGEST RESIDUAL MOMENT            2.090E-14   AT NODE        107   DOF  6
   INSTANCE: INST                                                                            
 LARGEST INCREMENT OF ROTATION      1.139E-03   AT NODE         26   DOF  6
   INSTANCE: INST                                                                            
 LARGEST CORRECTION TO ROTATION     1.248E-06   AT NODE        338   DOF  6
   INSTANCE: INST                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED   1.00    
 STEP TIME COMPLETED        1.00    ,  TOTAL TIME COMPLETED         1.00    


          THE ANALYSIS HAS BEEN COMPLETED



     ANALYSIS SUMMARY:
     TOTAL OF        100  INCREMENTS
                       0  CUTBACKS IN AUTOMATIC INCREMENTATION
                     116  ITERATIONS INCLUDING CONTACT ITERATIONS IF PRESENT
                     116  PASSES THROUGH THE EQUATION SOLVER OF WHICH 
                      42  INVOLVE MATRIX DECOMPOSITION, INCLUDING
                       0  DECOMPOSITION(S) OF THE MASS MATRIX
                       1  REORDERING OF EQUATIONS TO MINIMIZE WAVEFRONT
                       0  ADDITIONAL RESIDUAL EVALUATIONS FOR LINE SEARCHES
                       0  ADDITIONAL OPERATOR EVALUATIONS FOR LINE SEARCHES
                       1  WARNING MESSAGES DURING USER INPUT PROCESSING
                      84  WARNING MESSAGES DURING ANALYSIS
                      84  ANALYSIS WARNINGS ARE NUMERICAL PROBLEM MESSAGES
                       0  ANALYSIS WARNINGS ARE NEGATIVE EIGENVALUE MESSAGES
                       0  ERROR MESSAGES



     JOB TIME SUMMARY
       USER TIME (SEC)      =   5.4000    
       SYSTEM TIME (SEC)    =  0.90000    
       TOTAL CPU TIME (SEC) =   6.3000    
       WALLCLOCK TIME (SEC) =         11
