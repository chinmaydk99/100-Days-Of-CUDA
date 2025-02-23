Frontier with privatization workflow

START BFS → Initialize shared memory (per block) → 
Each thread processes vertices → 
Add new vertices to block-private shared memory → 
Synchronize → Commit shared frontier to global memory → 
Update BFS level → Repeat until all vertices are visited → END
