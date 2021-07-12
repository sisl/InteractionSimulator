# test_animate.py

from intersim.utils import get_map_path, get_svt, SVT_to_simstates
from intersim.viz import animate
import os
opj = os.path.join
def test_animate(graph=False):
    outdir  = opj('tests','output')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    nframes = 400

    # load a trackfile
    svt, svt_path = get_svt()
    osm = get_map_path()
    print('SVT path: {}'.format(svt_path))
    print('Map path: {}'.format(osm))
    
    states = SVT_to_simstates(svt)[:nframes]

    if graph:
        filestr = opj(outdir,'test_animate_graph')
        from intersim.graphs import ConeVisibilityGraph
        cvg = ConeVisibilityGraph(r=20, half_angle=120)
        graphs = []
        for i in range(len(states)):
            s = states[i]
            cvg.update_graph(s.reshape((-1,5)))
            graphs.append(cvg.edges)
    else:
        filestr = opj(outdir,'test_animate')
        graphs = None

    animate(osm, states, svt._lengths, svt._widths, graphs=graphs, filestr=filestr)

def test_animate_graph():
    pass

if __name__ == '__main__':
    test_animate()
    test_animate(graph=True)