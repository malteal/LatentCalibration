"visualize set/graphs/point cloud"
import matplotlib.pyplot as plt
import torch

def visualize_points(pos, edge_index=None, index=None, ax=None, style={}, nr =None):
    if isinstance(pos, torch.Tensor):
        pos = pos.cpu().detach().numpy()
    pos = pos[(pos != -999).any(1)]
    scatter = None
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(4, 4))
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
             src = pos[src].tolist()
             dst = pos[dst].tolist()
             ax.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
    if index is None:
        if nr is None:
            scatter = ax.scatter(pos[:, 0], pos[:, 1], **style)
        else:
            scatter = ax.scatter(pos[:, 0], np.ones_like(pos[:, 0])*nr, **style)
    else:
       mask = torch.zeros(pos.size(0), dtype=torch.bool)
       mask[index] = True
       ax.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray',
                  zorder=1000, **style)
       ax.scatter(pos[mask, 0], pos[mask, 1], **style)
    return ax, scatter