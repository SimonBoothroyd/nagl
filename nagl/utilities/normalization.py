"""SMARTS patterns used to normalize molecules.

These were taken directly from the RDKit [`rdkit/Code/GraphMol/MolStandardize/TransformCatalog/normalizations.in`](https://github.com/rdkit/rdkit/blob/67aa9f9062779500b87e774019d45ff19c34aba5/Code/GraphMol/MolStandardize/TransformCatalog/normalizations.in)
file at commit `67aa9f9` (see the `LICENSE-3RD-PARTY` file for more details), which
contained the following license header:

//
//  Copyright (C) 2021 Greg Landrum
//
//   @@ All Rights Reserved @@
//  This file is part of the RDKit.
//  The contents are covered by the terms of the BSD license
//  which is included in the file license.txt, found at the root
//  of the RDKit source tree.
//
"""

NORMALIZATION_SMARTS = [
    {
        "smarts": "[N,P,As,Sb;X3:1](=[O,S,Se,Te:2])=[O,S,Se,Te:3]>>[*+1:1]([*-1:2])=[*:3]",
        "comment": "Nitro to N+(O-)=O",
    },
    {
        "smarts": "[S+2:1]([O-:2])([O-:3])>>[S+0:1](=[O-0:2])(=[O-0:3])",
        "comment": "Sulfone to S(=O)(=O)",
    },
    {
        "smarts": "[nH0+0:1]=[OH0+0:2]>>[n+:1][O-:2]",
        "comment": "Pyridine oxide to n+O-",
    },
    {
        "smarts": "[*:1][N:2]=[N:3]#[N:4]>>[*:1][N:2]=[N+:3]=[N-:4]",
        "comment": "Azide to N=N+=N-",
    },
    {
        "smarts": "[*:1]=[N:2]#[N:3]>>[*:1]=[N+:2]=[N-:3]",
        "comment": "Diazo/azo to =N+=N-",
    },
    {
        "smarts": "[!O:1][S+0;X3:2](=[O:3])[!O:4]>>[*:1][S+1:2]([O-:3])[*:4]",
        "comment": "Sulfoxide to -S+(O-)-",
    },
    {
        "smarts": "[O,S,Se,Te;-1:1][P+;D4:2][O,S,Se,Te;-1:3]>>[*+0:1]=[P+0;D5:2][*-1:3]",
        "comment": "Phosphate to P(O-)=O",
    },
    {
        "smarts": "[C,S&!$([S+]-[O-]);X3+1:1]([NX3:2])[NX3!H0:3]>>[*+0:1]([N:2])=[N+:3]",
        "comment": "C/S+N to C/S=N+",
    },
    {
        "smarts": "[P;X4+1:1]([NX3:2])[NX3!H0:3]>>[*+0:1]([N:2])=[N+:3]",
        "comment": "P+N to P=N+",
    },
    {
        "smarts": "[CX4:1][NX3H:2]-[NX3H:3][CX4:4][NX2+:5]#[NX1:6]>>[CX4:1][NH0:2]=[NH+:3][C:4][N+0:5]=[NH:6]",
        "comment": "Normalize hydrazine-diazonium",
    },
    {
        "smarts": "[N,P,As,Sb,O,S,Se,Te;-1:1]-[A+0:2]=[N,P,As,Sb,O,S,Se,Te;+1:3]>>[*-0:1]=[*:2]-[*+0:3]",
        "comment": "Recombine 1,3-separated charges",
    },
    {
        "smarts": "[n,o,p,s;-1:1]:[a:2]=[N,O,P,S;+1:3]>>[*-0:1]:[*:2]-[*+0:3]",
        "comment": "Recombine 1,3-separated charges",
    },
    {
        "smarts": "[N,O,P,S;-1:1]-[a:2]:[n,o,p,s;+1:3]>>[*-0:1]=[*:2]:[*+0:3]",
        "comment": "Recombine 1,3-separated charges",
    },
    {
        "smarts": "[N,P,As,Sb,O,S,Se,Te;-1:1]-[A+0:2]=[A:3]-[A:4]=[N,P,As,Sb,O,S,Se,Te;+1:5]>>[*-0:1]=[*:2]-[*:3]=[*:4]-[*+0:5]",
        "comment": "Recombine 1,5-separated charges",
    },
    {
        "smarts": "[n,o,p,s;-1:1]:[a:2]:[a:3]:[c:4]=[N,O,P,S;+1:5]>>[*-0:1]:[*:2]:[*:3]:[c:4]-[*+0:5]",
        "comment": "Recombine 1,5-separated charges",
    },
    {
        "smarts": "[N,O,P,S;-1:1]-[c:2]:[a:3]:[a:4]:[n,o,p,s;+1:5]>>[*-0:1]=[c:2]:[*:3]:[*:4]:[*+0:5]",
        "comment": "Recombine 1,5-separated charges",
    },
    {
        "smarts": "[N,O;+0!H0:1]-[A:2]=[N!$(*[O-]),O;+1H0:3]>>[*+1:1]=[*:2]-[*+0:3]",
        "comment": "Normalize 1,3 conjugated cation",
    },
    {
        "smarts": "[n;+0!H0:1]:[c:2]=[N!$(*[O-]),O;+1H0:3]>>[*+1:1]:[*:2]-[*+0:3]",
        "comment": "Normalize 1,3 conjugated cation",
    },
    {
        "smarts": "[N,O;+0!H0:1]-[A:2]=[A:3]-[A:4]=[N!$(*[O-]),O;+1H0:5]>>[*+1:1]=[*:2]-[*:3]=[*:4]-[*+0:5]",
        "comment": "Normalize 1,5 conjugated cation",
    },
    {
        "smarts": "[n;+0!H0:1]:[a:2]:[a:3]:[c:4]=[N!$(*[O-]),O;+1H0:5]>>[n+1:1]:[*:2]:[*:3]:[*:4]-[*+0:5]",
        "comment": "Normalize 1,5 conjugated cation",
    },
    {
        "smarts": "[F,Cl,Br,I,At;-1:1]=[O:2]>>[*-0:1][O-:2]",
        "comment": "Charge normalization",
    },
    {
        "smarts": "[N,P,As,Sb;-1:1]=[C+;v3:2]>>[*+0:1]#[C+0:2]",
        "comment": "Charge recombination",
    },
]
