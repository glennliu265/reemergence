#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:12:30 2024

@author: gliu
"""

outdict = scm.integrate_entrain(smconfig['h'],smconfig['kprev'],smconfig['lbd_a'],smconfig['forcing'],
                                Tdexp=smconfig['lbd_d'],beta=smconfig['beta'],add_F=smconfig['add_F'],
                                return_dict=True,old_index=True,Td_corr=smconfig['Td_corr'])



h                   = smconfig['h']
kprev               = smconfig['kprev']
lbd_a               = smconfig['lbd_a']
F                   = smconfig['forcing']


Tdexp               = smconfig['lbd_d']
beta                = smconfig['beta']
return_dict         = True
old_index           = True
add_F               = smconfig['add_F']
Td_corr             = smconfig['Td_corr']

multFAC = True
Td0     = False



# From inside the function
nlon,nlat,ntime = F.shape

# Replace T0 if it is not set
if T0 is None:
    T0 = np.zeros((nlon,nlat))

# Calculate beta
if beta is None:
    beta = calc_beta(h)

# Add entrainment damping, set up lbd terms
lbd = lbd_a + beta
#explbd = np.exp(-lbd)

# Calculate Integration Factor
FAC = 1
if multFAC:
    FAC = scm.calc_FAC(lbd)
    
# Set Td damping timescale
if Tdexp is None:
    Tdexp = np.zeros(FAC.shape)
    if Td_corr is True:
        Tdexp = np.ones(FAC.shape) # Since we are not exponentiating, Tdexp * T = T if Tdexp = 1



# Preallocate
T            = np.zeros((nlon,nlat,ntime))
damping_term = T.copy()
forcing_term = T.copy()
entrain_term = T.copy()
Td           = T.copy()

# Loop for each point

o = 0
a = 0
    
    
                
# Skip land/ice points, checking the forcing term
if np.any(np.isnan(F[o,a,:])):
    print("NaN Detected")
    #continue
 
# Get Last [Tdgrab] values for selected point
if Td0 is not False:
    Tdgrab = Td0[o,a,:]
else:
    Tdgrab = None
    
# If initial values are provided, get value
T0_in = 0#T0[o,a]

# Check for additional forcing
if add_F is not None:
    add_F_in = add_F[o,a,:]
else:
    add_F_in = None
    
# Integrate in time
temp_ts,damp_ts,noise_ts,entrain_ts,Td_ts = scm.entrain(ntime,lbd[o,a,:],T0_in,F[o,a,:],beta[o,a,:],h[o,a,:],kprev[o,a,:],FAC[o,a,:],
                                                    multFAC=multFAC,debug=True,debugprint=False,Tdgrab=Tdgrab,add_F=add_F_in,Tdexp=Tdexp[o,a,:],
                                                    old_index=old_index,Td_corr=Td_corr)

#%%
entrain(t_end,lbd,T0,F,beta,h,kprev,FAC,multFAC=1,debug=False,debugprint=False,
            Tdgrab=None,add_F=None,return_dict=False,Tdexp=None,old_index=False,Td_corr=False)

Tdgrab=None

multFAC = multFAC

t_end = ntime
lbd = lbd[o,a,:]
T0_in = T0_in
F = F[o,a,:]
beta = beta[o,a,:]
h = h[o,a,:]
kprev = kprev[o,a,:]
FAC = FAC[o,a,:]
Tdexp = Tdexp[o,a,:]

debugprint=False
#%%

T0 = T0.squeeze()


# Preallocate
temp_ts = np.zeros(t_end) * T0.squeeze()

# If Tdgrab is on, Grab the temperatures
if Tdgrab is None:
    t0  = 0 # Start from 0
    Td0 = None
else: # New length is t_end + len(Tdgrab)
    t0  = len(Tdgrab)
    Td0 = None # Set to this initially
    t_end += len(Tdgrab) # Append years to beginning of simulation
    temp_ts = np.concatenate([Tdgrab,temp_ts])
    
if Tdexp is None:
    Tdexp = np.zeros(12)
    
if np.any(np.isnan(Tdexp)):
    print("Warning, temporary correct, converting NaNs in lbd_d to zero")
    Tdexp[np.isnan(Tdexp)] = 0
    print("\t need to check interpolating/lbd_d estimation step")
    print("\t this will turn off entrainment at these steps")
    #print("This will effectively elimiated")
    
    
if debug:
    noise_ts   = np.zeros(t_end)
    damp_ts    = np.zeros(t_end)
    entrain_ts = np.zeros(t_end)
    Td_ts      = np.zeros(t_end)
    
entrain_term = np.zeros(t_end)

# Prepare the entrainment term
explbd = np.exp(np.copy(-lbd))
#explbd[explbd==1] = 0

# Loop for integration period (indexing convention from matlab)
for t in np.arange(t0,t_end,1):
    # if t == 18:
    #     break
    # Get the month (start from Jan, so +1)
    im  = t%12
    m   = im+1
    
    # --------------------------
    # Calculate entrainment term
    # --------------------------
    if (t<12) and (Tdgrab is None): # Only start Entrainment term after first 12 months
        if debugprint:
            print("Skipping t=%i" % t)
        entrain_term = 0
    else:
        
        if beta[im] == 0:       # For months with no entrainment
            entrain_term = 0    # Set Entrainment Term to Zero
            Td0          = None # Reset Td0 term
            if debugprint:
                print("No entrainment on month %i"%m)
                print("--------------------\n")
        else: # Retrieve temperature at detraining months
            
        
            
        
            if (Td0 is None) & (h.argmin()==im-1) :# For first entraining month
                Td1 = scm.calc_Td(t,kprev,temp_ts,prevmon=False,debug=debugprint)
                Td0 = Td1 # Previous month does not have entrainment!
            
            if (Td0 is None): # Calculate Td0 for other entraining months
                Td1,Td0 = scm.calc_Td(t,kprev,temp_ts,prevmon=True,debug=debugprint)
            else: # Use Td0 from last timestep
                Td1 = scm.calc_Td(t,kprev,temp_ts,prevmon=False,debug=debugprint)
            
            # Now apply deep damping
            if (Td0 is None): # For cases where there is no Td0
                
                if (h.argmin()==im-1): # For first entraining month
                    if Td_corr: # Conversion not needed (direct correlation provided)
                        decay_factor = Tdexp[m-1]
                        
                    else:
                        delta_t_1       = scm.calc_kprev_dmon(m,kprev) # dt = current month - detraining month
                        decay_factor    = np.exp(-Tdexp[m-1] * delta_t_1)
                    Td1             = Td1 * decay_factor
                    Td0             = Td1 # Previous month does not have entrinment
                    
                else: # Calculate Td0 (NOTE NEED TO CHECK THIS STEP, when might this apply? It seems to not really apply)
                    print("Td0=None condition applies on month %i" % m)
                    print("Exiting script, check scm line 833 (or around here).")
                    #break
                    if Td_corr:
                        delta_t_1     = scm.calc_kprev_dmon(m,kprev)
                        decay_factor1 = np.exp(-Tdexp[m-1] * delta_t_1)
                        delta_t_0     = scm.calc_kprev_dmon(m-1,kprev)
                        decay_factor0 = np.exp(-Tdexp[m-2] * delta_t_0)
                    else:
                        decay_factor1 = Tdexp[m-1]
                        decay_factor0 = Tdexp[m-2]
                    
                    Td1           = Td1 * decay_factor1
                    Td0           = Td0 * decay_factor0
                
            else: # Use Td0 from last timestep
                if Td_corr:
                    decay_factor = Tdexp[m-1]
                else:
                    delta_t_1    = scm.calc_kprev_dmon(m,kprev)
                    decay_factor = np.exp(-Tdexp[m-1] * delta_t_1)
                Td1          = Td1 * decay_factor

            # Compute Td (in future, could implement month-dependent decay..)
            Td = (Td1+Td0)/2
            if debugprint:
                print("Td is %.2f, which is average of Td1=%.2f, Td0=%.2f"%(Td,Td1,Td0)) 
                print("--------------------\n")
            Td0 = np.copy(Td1)# Copy Td1 to Td0 for the next loop
            
            # Calculate entrainment term
            entrain_term = beta[m-1]*Td
    
    # ----------------------
    # Get Noise/Forcing Term
    # ----------------------
    if old_index:
        t_get = t
    else:
        t_get = t-1
    noise_term = F[t_get]
    if add_F is not None:
        noise_term = F[t_get] + add_F[t_get]
    
    # ----------------------
    # Calculate damping term
    # ----------------------
    if t == 0:
        damp_term = explbd[im]*T0
    else:
        damp_term = explbd[im]*temp_ts[t-1]
    
    # ------------------------
    # Check Integration Factor
    # ------------------------
    if multFAC:
        integration_factor = FAC[im]
    else:
        integration_factor = 1
    
    
    # -----------------------
    # Compute the temperature
    # -----------------------
    temp_ts[t] = damp_term + (noise_term + entrain_term) * integration_factor

    # ----------------------------------
    # Save other variables in debug mode
    # ----------------------------------
    if debug:
        damp_ts[t]    = damp_term
        noise_ts[t]   = noise_term * integration_factor
        entrain_ts[t] = entrain_term * integration_factor
