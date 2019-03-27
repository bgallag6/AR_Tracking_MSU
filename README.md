# AR_Tracking_MSU

lets see if this works -- it did work.

ok, some time has passed.  will this work?

|      IDL                             |    Python                                                    | 
| ------------------------------------ | ------------------------------------------------------------ |
| stdev(arr)                           | np.std(arr, ddof=1)                                          |
| Y = (shdr.lebxsum>1)                 | y = max(shdr['lebxsum'], 1)                                  |
| Rotate(x, N)                         | 1.) t, n = divmod(N, 4)                                      |
|                                      | 2) if t = 1: np.rot90(x.T, -n) ; if t = 0: np.rot90(x, -n)]  |
| a # b                                | np.matmul(a, b) -or- np.matmul(a.T, b.T).T                   |
| STRMID(str0, a, b, /reverse_offset)  | str0[-(a+1) : -(a+1)+b]                                      |
| STRMID(str0, a, b)                   | str0[a : a+b]                                                |
| STRMID(str0, a, STRLEN(str0)-a)      | str0[a:]                                                     |
| STRLEN(str0)                         | len(str0)                                                    |
| TOTAL(x, 2)                          | x.sum(axis=0)                                                |      
| FIX(x)                               | int(x)                                                       |
| y = INTARR(2, 2)                     | y = np.zeros((2, 2), dtype=int)                              |
| DELVARX, temp                        | del temp                                                     |
| STRPOS(str0, 'a')                    | str0.find('a')                                               |
| STRPUT, str0, 'one', 5               | slen = len('one') ; str0.replace(str0[5:5+slen], 'one')      |
| REFORM(arr, 2, 3)                    | arr.reshape(3, 2)                                            |

IDL:
```idl
IF (hdr.detector EQ 'C2') OR (hdr.detector EQ 'C3') OR (hdr.detector EQ EIT) THEN BEGIN
```
Python:
```python
if hdr['detector'] in ['C2', 'C3', 'EIT']:
```

IDL:
```idl
w = WHERE(data gt 3, count, complement=wb, ncomplement=ncount)
```
Python:
```python
w = np.where(data > 3) ; count = len(w[0]) ; ncount = len(data) - count
def complement(arr, where):        
    y = np.indices((arr.shape[0], arr.shape[1]))
            
    y3, z3, cmplmnt = [], [], []       
    for z0, z1 in zip(where[0], where[1]):
        z3.append([z0,z1])
    
    for y0, y1 in zip(y[0].flatten(), y[1].flatten()):
        y3.append([y0,y1])        
    for y0 in y3:
        if y0 not in z3:
            cmplmnt.append(y0)
    
    cmplmnt = tuple((np.array(cmplmnt)[:,0], np.array(cmplmnt)[:,1]))
    return cmplmnt
```

IDL:
```idl
VALID_NUM(str0)
```
Python:
```python
def valid_num(str0):
    try:
        float(str0)
        return True
    except ValueError: 
        return False
valid_num(str0)
```

