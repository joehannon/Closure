
**Closure**

Work in progress on research and development of closure models for fast reactions in turbulent liquids, especially expressions for C(f).

Not ready to be used widely yet. Only take a copy of you really know what you are doing :) I will make this pip installable when it's fully ready for wider use, probably Q4 2026.

The key code is in species_limits.py.  That reads a JSON describing the chemical reaction system:
```
python3 species_limits.py inputs/<filename>.JSON
```

You can get the gist of what this does by reading my [blog](https://joehannon.github.io/blog/).

There's also a preprint available at [ChemRxiv](https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15006522/v1).

The following videos show the ray limit method for estimating the infinitely fast reaction limit:

Limits changing with time during a simulation:

[Limits changing with time during a simulation](https://github.com/user-attachments/assets/fad753fd-be26-426c-bf49-5c9d27e62d63)

Limits changing with mixing intensity when we sweep over a range of $\epsilon$ values (W/kg) from low to high:

[Limits changing with mixing intensity when we sweep over a range of $\epsilon$ values (W/kg) from low to high](https://github.com/user-attachments/assets/df506b03-5ce7-4044-b82f-17da84cabd09)

