# Closure #

Work in progress on research and development of closure models for fast reactions in turbulent liquids, especially expressions for C(f) via a new closure for the infinitely fast limit, B(f).

The code is not ready to be used widely yet. Only take a copy if you really know what you are doing :) I will make the code pip installable when it's fully ready for wider use, planned for Q4 2026.

The program code is all in species_limits.py.  That reads a JSON describing the chemical reaction system:
```
python3 species_limits.py inputs/<filename>.JSON
```

You can get the gist of what this does by reading my [blog](https://joehannon.github.io/blog/).  There's also a preprint available at [ChemRxiv](https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15006522/v1).

## Videos ##
The following videos illustrate the dynamic nature of the ray limit method for estimating the infinitely fast reaction limit.

**Videos in format used in the latest code, "species view"**

Limits $B(f)$ changing with time during a simulation, $\epsilon$=1 W/kg:

https://github.com/user-attachments/assets/b56b4471-ff4a-4420-a6fe-34337f5ea9bd

$C(f)$ moving towards $B(f)$ at $\epsilon$=1E6 W/kg:

https://github.com/user-attachments/assets/dcf397f9-4ed5-4f9b-9312-c2ba2f42d4bd

Limits changing with mixing intensity when we sweep over a range of $\epsilon$ values from 1E-6 to 1E6 W/kg:

https://github.com/user-attachments/assets/91dd986e-f1ff-404e-a259-d857f799b23d

**Videos in format used in the ChemRxiv preprint, "subset view"**

Limits changing with time during a simulation:

[Limits changing with time during a simulation](https://github.com/user-attachments/assets/fad753fd-be26-426c-bf49-5c9d27e62d63)

Limits changing with mixing intensity when we sweep over a range of $\epsilon$ values (W/kg) from low to high:

[Limits changing with mixing intensity when we sweep over a range of $\epsilon$ values (W/kg) from low to high](https://github.com/user-attachments/assets/df506b03-5ce7-4044-b82f-17da84cabd09)

