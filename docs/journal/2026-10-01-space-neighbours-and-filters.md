# 2026-10-01: Ranking and filtering an item's neighbours in the Space

## Current State

### Ranking

- Scoring neighbours over the item's 8 heaviest features only put Lockheed Martin F-22 Raptor first for French Bulldog (Q29149) and fiscal federalism first for the Kalman filter; scoring over all the item's features gives Golden Retriever, Persian cat, Rottweiler and Chow Chow, and random walk 0.79, urban economics 0.78, elliptic function 0.78, as sae/neighbours.py does — reading all the Kalman filter's 13 features took 11.8 s and 36 MB, and its 10 features on 250k items or fewer 1.6 s.
- space/data.js `neighbours` scores over up to 16 of the item's features on 250k items or fewer, and ranks every candidate once; filters pick from all of them — a version that kept the top 2,000 before filtering showed superintelligence (Q1566000) no neighbour of its own kind.
- French Bulldog (Q29149) has 20 external-ID properties, and the American Kennel Club ID (P13890, 283 items) is its only dog-specific one; none of its features in v0 is specific to dogs.
- attention (Q103701642) has 2 external IDs and in v0 2 features, only Encyclopedia of China (51,537 items) under 250k items, so its neighbours scored 0.95 alike; space/index.html says so above the list when one or no feature is on 250k items or fewer.

### Kinds

- space/index.html shows each neighbour's kinds ("instance of", or for a class its "subclass of" parents), in orange when none is under the item's kinds or the classes up to 2 steps above them.
- "Only:" buttons offer the item's kinds and the classes up to 3 steps above (short of top-level classes such as entity, object and concept); several can be chosen (all must hold); "Same kind by default" (kept in `localStorage`) starts with the item's first kind — the Kalman filter's neighbours become algorithms, the French Bulldog's dog breeds, the red fox's taxa.
- With a filter on, the 8 nearest items it leaves out follow greyed under a rule.
- "under this type (N)" lists a type's instances and subclasses from `members.parquet`, 3 levels down (at most 500), by similarity, those sharing no feature after at 0 (a fake publish: 299 instances, 11 subclasses and 10 instances of a subclass gave 320).

### Pinned statements

- Bulbasaur's first generation is a statement ("first appearance": first generation of Pokémon, P4584), not a kind; a + beside each value in the Wikidata links panel pins its statement, checked on Wikidata (`wbgetentities` claims, 50 ids a request, cached) for the most similar items of the chosen kinds — Bulbasaur with starter Pokémon and that pin gives Eevee, Squirtle, Charmander and Pikachu.
- Widening the kinds dropped a match when only the top 200 were checked (superintelligence, "subclass of: artificial general intelligence": friendly artificial intelligence was among 21 of kind "type of intelligence", outside the 200 most similar of any kind); items already checked now count wherever they rank.
- For PowerShell (Q840410) with "developer: Microsoft" pinned and "Same kind by default" then unchecked, the progress restarted at each 200-item round and two updates ran at once, the earlier writing last; the page now checks the 200 most similar, then the rest up to 1,000 in one step, with one rising count ("Checked N of up to 1,000 most similar on Wikidata · M matches so far"), and only the latest update writes (a generation counter) — ending with Visual Studio Code, Sticky Notes, Microsoft Launcher and Windows 1.0.
