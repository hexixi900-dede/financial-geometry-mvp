# ChartAgent method-source audit

- Official ACL record: https://aclanthology.org/2026.acl-long.843/
- Official arXiv record/source: https://arxiv.org/abs/2510.04514
- Checked on: 2026-08-15/16 (Asia/Shanghai)

No author-published executable ChartAgent repository was found in the ACL record, arXiv source, paper bibliography, author-linked paper pages, or GitHub repository search. The arXiv source includes Python-styled signatures, docstrings, arguments, return values, and examples for the following relevant tools, but not function bodies:

- `axis_localizer(image, axis, axis_threshold, axis_tickers)`
- `interpolate_pixel_to_value(pixel, axis_values, axis_pixel_positions)`
- `get_bar(image, rgb_of_interest, ticker_label, segmentation_model, bar_orientation)`
- `compute_bar_height(...)`
- `get_edgepoints(...)`

This MVP treats those appendix definitions as a behavioral specification and independently reimplements only the CPU geometry subset. It does not recreate AutoGen orchestration, SAM/Semantic-SAM routing, visual self-refinement loops, complete QA, or other chart types.
