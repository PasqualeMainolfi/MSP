## **MSP**  
### *Sound Processing via Matching Pursuit with Time-Frequency Dictionary*

Python implementation of a Matching Pursuit algorithm discusses in *Matching Pursuit with Time-Frequency Dictionary*, S. G. Mallat, Z. Zhang, IEEE transaction 1993, for sound processing.

1. Import

```python
from utils.matching_sound_processing import MSP
```
2. Define *MP* object, specifying target path, source path and sample rate.

```python
mp = MSP(target_path=TARGET, source_path=SOURCE, sr=SR)
```

3. Create atoms and time-frequency dictionary, specifying:
   - decomposition mode: str, fixed or variable. 
   - The argumets associated: if fixed, you will have to define *wlen* (win length in samples) and *hopsize* (in percent = hopsize * wlen); if varbiale, you will to define *wlenmin, wlenmax, hopsizemin, hopsizemax* and number of lengths to generate, nwin

```python
mp.generate_atoms(mode="variable", wlenmin=1024, wlenmax=4096, hopsizemin=0.25, hopsizemax=3, nwin=10)
```

4. Generate matching atoms, specifying: the max number of atoms to be extract *k* and the max error *eps* 

```python
mp.matching(k=10, eps=1e-6)
```

5. rebuild  target signal (return 1D array)

```python
y = mp.perform_rebuild()
```  

It is possibile access to all parameters:


```python
# target frames
target_atoms = mp.matching_pursuit.target_atoms

# dictionary
dictionary = mp.matching_pursuit.dictionary

# coeffs with best index 
coeffs = mp.matching_pursuit.coeffs

# atoms with best index 
atoms = mp.matching_pursuit.atoms

# result of the product between coeffs and atoms with best index (2D)
matching_atoms = mp.matching_pursuit.matching_atoms

```

...and plotting the results


```python
mp.plot_results()

```