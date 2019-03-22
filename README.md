# AR_Tracking_MSU

lets see if this works -- it did work.

ok, some time has passed.  will this work?

|      IDL              |    Python                                                  | 
| -------------         | ---------------------------------------------------------- |
| stdev(arr)            | np.std(arr, ddof=1)                                        |
| Y = (shdr.lebxsum>1)  | y = max(shdr['lebxsum'], 1)                                |
| Rotate(x, N)          | 1.) n, t = divmod(N, 4)                                    |
|                       | 2) if t = 1: np.rot90(x, n).T ; if t = 0: np.rot90(x, n)]  |

IDL:
```idl
IF (hdr.detector EQ 'C2') OR (hdr.detector EQ 'C3') OR (hdr.detector EQ EIT) THEN BEGIN
```
Python:
```python
if hdr['detector'] in ['C2', 'C3', 'EIT']:
```

