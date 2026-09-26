# Closure #

Research and development of closure models for complex systems of fast reactions in turbulent liquids, especially expressions for $B(f)$, the complete/ infinitely fast reaction limit that bounds $C(f)$.

The code illustrates closures and will soon be ready to release for application to user-defined reaction systems. See the [PYPI_README](PYPI_README.md) and [JSON_OPTIONS](JSON_OPTIONS.md) for additional information. The code will be made pip installable from PyPi, planned for Q4 2026.

The code is currently all in species_limits.py.  That reads a JSON describing the chemical reaction system from an inputs subfolder; results appear in several additional subfolders, including plots and lines:
```
python3 species_limits.py examples/<filename>.JSON
```

If the JSON lives in a folder named "inputs", outputs go one level up — a sibling of inputs/. If the JSON is anywhere else, outputs land right next to it.

You can get more context and detail by reading my [blog](https://joehannon.github.io/blog/).  There's also a preprint available at [ChemRxiv](https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15006522/v1) and a journal publication will be available soon.

## Videos ##
The following videos illustrate the dynamic nature of the ray limit method for estimating the infinitely fast reaction limit. Each one features a 5-reaction version of the second Bourne reaction, an azo-coupling that is mixing-sensitive. These are also available on [YouTube](https://www.youtube.com/watch?v=-xXvNcTVq4A&list=PLdw4xLy1H_i0).

**Videos in format used in the latest code, "species view"**

Species view shows the static limits that may be pre-calculated for each of the $2^N$ infinitely fast reaction subsets in an $N$-reaction system.  Ray limit's $B(f)$ is overlaid (black squares) on a subplot for each species.

Limits $B(f)$ changing with time during a simulation, $\epsilon$=1E-6 W/kg:

https://github.com/user-attachments/assets/4f61df09-af7f-4f99-ad16-25320d037b8b

$C(f)$ moving towards $B(f)$ at $\epsilon$=1E6 W/kg:

https://github.com/user-attachments/assets/29f8f23c-0474-411b-a2f5-859120aa4a00

Limits $B(f)$ changing with mixing intensity when we sweep over a range of $\epsilon$ values from 1E-6 to 1E6 W/kg:

https://github.com/user-attachments/assets/b828b8f2-f5d7-4980-aa9e-4a7fde783f85

Heatmap of $B(f)$ versus f and time when we sleep over a range of $\epsilon$ values from 1E-6 to 1E6 W/kg:

https://github.com/user-attachments/assets/fa8ba5f8-dae1-4737-aa5f-59046f50a383

**Videos in format used in the ChemRxiv preprint, "subset view"**

Subset view shows ray limit's automatically calculated $B(f)$ on a single plot showing all species. Here the final pdf of f has been added to the plot.

Limits changing with mixing intensity when we sweep over a range of $\epsilon$ values (W/kg) from low to high:

https://github.com/user-attachments/assets/e5225f6e-fb57-4bce-a863-238e0fb98860

