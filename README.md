## **MSP**  
### *Sound Processing via Matching Pursuit with Time-Frequency Dictionary*

Python implementation of a Matching Pursuit algorithm discusses in *Matching Pursuit with Time-Frequency Dictionary*, S. G. Mallat, Z. Zhang, IEEE transaction 1993, for sound processing.

```python
from utils.mp import MP

mp = MP(target_path=TARGET, source_path=SOURCE, sr=SR)

# create atoms and time-freq dictionary
# mode: fixed (win and hop with fixed size) or variable (win and hop with variable size)
mp.generate_atoms(mode="variable", wlenmin=1024, wlenmax=4096, hopsizemin=0.25, hopsizemax=3, n_win=10)

# generaete matching atoms
mp.matching(k=10, eps=1e-6)

# rebuild target -> 1D array
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

# result of the product between coeffs and atoms with best index
matching_atoms = mp.matching_pursuit.matching_atoms

```


and plotting the results


```python
mp.plot_results()

```