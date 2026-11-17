# CCTV600 parsing study

This experiment owns the nine-checkpoint CCTV600 RGB-crop parsing benchmark.
Keep dataset bytes, prior detector/native/CCTV90 experiments and shared adapters immutable.
Private images, annotations, per-image identifiers, packets and logs stay in outputs/.
Commit source, protocol, documentation and reviewed aggregate evidence only.
Unassigned GT is unknown. Never turn it into background or infer absent parts.
The annotation-supported domain is a diagnostic comparison, with full-frame scores null.
Freeze bindings before inference. Sequential workers, atomic packets, zero model retries.
Resource admission and clean timing checks apply independently of semantic quality.
Analyze quality once the approved batch stops. Preserve failures and partial packets.
