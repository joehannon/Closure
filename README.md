# Closure #

Research and development of closure models for fast reactions in turbulent liquids, especially expressions for $B(f)$, the infinitely fast reaction limit that bounds $C(f)$.

The code is not ready to be used widely yet. Only take a copy if you really know what you are doing :) The code will be made code pip installable for wider use, planned for Q4 2026.

The program code is all in species_limits.py.  That reads a JSON describing the chemical reaction system from an inputs subfolder; results appear in several additional subfolders, including plots and lines:
```
python3 species_limits.py examples/<filename>.JSON
```

If the JSON lives in folder names "inputs", outputs go one level up — a sibling of inputs/. If the JSON is anywhere else, outputs land right next to it.

You can get more context and detail by reading my [blog](https://joehannon.github.io/blog/).  There's also a preprint available at [ChemRxiv](https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15006522/v1) and a journal publication will be available soon.

## Videos ##
The following videos illustrate the dynamic nature of the ray limit method for estimating the infinitely fast reaction limit. Each one features a 5-reaction version of the second Bourne reaction, an azo-coupling that is mixing-sensitive.

**Videos in format used in the latest code, "species view"**

Species view shows the static limits that may be pre-calculated for each of the $2^N$ infinitely fast reaction subsets in an $N$-reaction system.  Ray limit's $B(f)$ is overlaid (black squares) on a subplot for each species.

Limits $B(f)$ changing with time during a simulation, $\epsilon$=1 W/kg:

https://github.com/user-attachments/assets/b56b4471-ff4a-4420-a6fe-34337f5ea9bd

$C(f)$ moving towards $B(f)$ at $\epsilon$=1E6 W/kg:

https://github.com/user-attachments/assets/dcf397f9-4ed5-4f9b-9312-c2ba2f42d4bd

Limits changing with mixing intensity when we sweep over a range of $\epsilon$ values from 1E-6 to 1E6 W/kg:

https://github.com/user-attachments/assets/91dd986e-f1ff-404e-a259-d857f799b23d

**Videos in format used in the ChemRxiv preprint, "subset view"**

Subset view shows only ray limit's automatically calculated $B(f)$ on a single plot showing all species.

Limits changing with time during a simulation:

[Limits changing with time during a simulation](https://github.com/user-attachments/assets/fad753fd-be26-426c-bf49-5c9d27e62d63)

Limits changing with mixing intensity when we sweep over a range of $\epsilon$ values (W/kg) from low to high:

[Limits changing with mixing intensity when we sweep over a range of $\epsilon$ values (W/kg) from low to high](https://github.com/user-attachments/assets/df506b03-5ce7-4044-b82f-17da84cabd09)

