# c3_callgraphs  

## Installation  

We require a static call graph analysis.  

Install `pycg`:  

```bash
$ pip install pycg
```

Module import errors due to capitalised `PyCG/` in miniconda3 site-packages - `mv PyCG pycg` solved it.

## Usage  

### Make json  

Generate .json for `cogent3.core.alignment`:

```bash
$ cd c3_callgraphs
$ python -m pycg --package Cogent3 ../Cogent3/src/cogent3/core/alignment.py -o alignment.json
```  

Remove leading full stops:
```bash
sed 's/......Cogent3.src.//g' alignment.json > alignment_stripped.json
```
### Visualise  

