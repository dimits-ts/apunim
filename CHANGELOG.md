# What's new

### 1.0.6 (24/08/2026)
- Fix DFU bin grid. `aposteriori_unimodality` now histograms every group and
  every random partition on one grid fixed over the full annotation scale.
  Previously each sample was binned over its *own* range, so a group that did
  not reach both ends of the scale acquired empty bins between populated ones
  and was scored as multimodal. Because the size-matched random partitions mix
  all groups and therefore do span the scale, this biased observed DFU upwards
  relative to the apriori baseline and could invert the sign of the apunim
  statistic for concentrated groups.
- `dfu` now accepts an explicit sequence of bin edges in addition to a bin
  count. Passing an integer keeps the previous, range-dependent behaviour.

### 1.0.4 (14/07/2026)
- Invert sign of apunim statistic

### 1.0.3 (25/06/2026)
- Fix bug causing program crashes when Nones were introduced in the input of the `aposteriori_unimodality` function.

### 1.0.2 (14/05/2026)
- Expose only public API
- Improve internal filtering of non-applicable comments

## 1.0.1 (5/12/2025)
- Added support in ApunimResult to indicate number of observations per factor
- Linked documentation in README for better accessibility.

## 1.0.0 (24/11/2025)
- Initial release