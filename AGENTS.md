# CCTV600 parsing study

This experiment owns the nine-checkpoint CCTV600 RGB-crop parsing benchmark.
Keep dataset bytes, prior detector/native/CCTV90 experiments and shared adapters immutable.
Private images, annotations, per-image identifiers, packets and logs stay in outputs/.
Commit source, protocol, documentation and reviewed aggregate evidence only.
Unassigned GT is unknown. Never turn it into background or infer absent parts.
The active protocol is configs/SHARED_REFERENCE_V1.json, authorized by the user's
restart request. Use the earlier shared-reference Mean-part IoU, Face-local,
mIoU3 (part-mean) and Macro4-local definitions. Accepted positive part masks define
reference extents for per-person-part tasks. Shared valid grid excludes explicit
ignore and QA conflicts. Source unassigned retains its dataset meaning.
Use bbox matching for CCTV600 and disclose its difference from GT-mask matching.
Historical AD4 outputs/config/source remain immutable evidence of the prior scope.
Freeze bindings before inference. Sequential workers, atomic packets, zero model retries.
Resource admission and clean timing checks apply independently of semantic quality.
Analyze quality once the approved batch stops. Preserve failures and partial packets.
