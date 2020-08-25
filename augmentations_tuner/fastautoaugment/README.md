# Fast AutoAugment on a single GPU

Based on the official [implementation of Fast AutoAugment](https://github.com/kakaobrain/fast-autoaugment)
 but runs on a single GPU, uses the newest version of Ray, and is much easier to get running.

- Fast AutoAugment learns augmentation policies using a more efficient search strategy based on density matching.
- Fast AutoAugment speeds up the search time by orders of magnitude while maintaining the comparable performances.

<p align="center">
<img src="etc/search.jpg" height=350>
</p>

