from intersim.viz import Rasta
import numpy as np

def test_world_to_raster():
    rasta = Rasta(
        m_per_px = 0.5,
        raster_fixpoint = (0.5, 0.5),
        world_fixpoint = (100, 100),
        camera_rotation = 0
    )
    coords = rasta._world_to_raster(
        raster_shape=(50, 50),
        points=np.array([[100, 100]]),
    )
    expect = np.array([[25, 25]])
    assert np.allclose(coords, expect)

def test_world_to_raster_multiple():
    rasta = Rasta(
        m_per_px = 0.5,
        raster_fixpoint = (0.5, 0.5),
        world_fixpoint = (100, 100),
        camera_rotation = 0
    )

    points = np.array([[[
        [100, 100],
        [100, 100],
        [100, 100],
        [100, 100],
    ]]])
    coords = rasta._world_to_raster(
        raster_shape=(50, 50),
        points=points,
    )

    expect = np.array([[[
        [25, 25],
        [25, 25],
        [25, 25],
        [25, 25],
    ]]])

    assert points.shape == expect.shape
    assert coords.shape == expect.shape
    assert np.allclose(coords, expect)

def test_world_to_raster_no_side_effects():
    rasta = Rasta(
        m_per_px = 0.5,
        raster_fixpoint = (0.5, 0.5),
        world_fixpoint = (100, 100),
        camera_rotation = 0
    )
    raster_shape = (50, 50)
    points = np.array([[100, 100]])
    rasta._world_to_raster(
        raster_shape=raster_shape,
        points=points,
    )

    assert rasta._m_per_px == 0.5
    assert np.allclose(rasta._raster_fixpoint, np.array([0.5, 0.5]))
    assert np.allclose(rasta._world_fixpoint, np.array([100, 100]))
    assert rasta._camera_rotation == 0
    assert raster_shape == (50, 50)
    assert np.allclose(points, np.array([[100, 100]]))

def test_world_to_raster_offset():
    rasta = Rasta(
        m_per_px=0.5,
        raster_fixpoint=(0.5, 0.5),
        world_fixpoint=(43, 21),
        camera_rotation=0
    )
    coords = rasta._world_to_raster(
        raster_shape=(50, 50),
        points=np.array([[43 + 10, 21 - 24]]),
    )
    expect = np.array([[50*0.5 + 10/0.5, 50*0.5 + 24/0.5]])
    assert np.allclose(coords, expect)

def test_world_to_raster_rotation():
    rasta = Rasta(
        m_per_px=0.5,
        raster_fixpoint=(0.5, 0.5),
        world_fixpoint=(43, 21),
        camera_rotation=np.pi/2
    )
    coords = rasta._world_to_raster(
        raster_shape=(50, 50),
        points=np.array([43 + 10, 21 - 24]),
    )
    expect = np.array([50*0.5 - 24/0.5, 50*0.5 + 10/0.5])
    assert np.allclose(coords, expect)

def test_fill_poly():
    rasta = Rasta(
        m_per_px=0.5,
        raster_fixpoint=(0.5, 0.5),
        world_fixpoint=(100, 100),
        camera_rotation=np.pi/2
    )
    canvas = rasta.fill_poly(
        canvas=np.zeros((200, 200)),
        vertices=np.array([[[90., 90.], [110., 90.], [110., 110.], [90., 110.]]])
    )
    assert (canvas[80:120, 80:120] == 1).all()
    assert (canvas[:80] == 0).all()
    assert (canvas[:, :80] == 0).all()
    assert (canvas[121:] == 0).all()
    assert (canvas[:, 121:] == 0).all()

def test_fill_poly_multiple():
    rasta = Rasta(
        m_per_px=0.5,
        raster_fixpoint=(0.5, 0.5),
        world_fixpoint=(100, 100),
        camera_rotation=np.pi/2
    )
    canvas = rasta.fill_poly(
        canvas=np.zeros((200, 200)),
        vertices=np.array([
            [[90., 90.], [110., 90.], [110., 110.], [90., 110.]],
            [[90., 90.], [110., 90.], [110., 110.], [90., 110.]],
        ])
    )
    assert (canvas[80:120, 80:120] == 1).all()
    assert (canvas[:80] == 0).all()
    assert (canvas[:, :80] == 0).all()
    assert (canvas[121:] == 0).all()
    assert (canvas[:, 121:] == 0).all()

def test_fill_poly_single():
    rasta = Rasta(
        m_per_px=0.5,
        raster_fixpoint=(0.5, 0.5),
        world_fixpoint=(100, 100),
        camera_rotation=np.pi/2
    )
    canvas = rasta.fill_poly(
        canvas=np.zeros((200, 200)),
        vertices=np.array([[90., 90.], [110., 90.], [110., 110.], [90., 110.]])
    )
    assert (canvas[80:120, 80:120] == 1).all()
    assert (canvas[:80] == 0).all()
    assert (canvas[:, :80] == 0).all()
    assert (canvas[121:] == 0).all()
    assert (canvas[:, 121:] == 0).all()

def test_fill_poly_array():
    rasta = Rasta(
        m_per_px=0.5,
        raster_fixpoint=(0.5, 0.5),
        world_fixpoint=(100, 100),
        camera_rotation=np.pi/2
    )
    canvas = rasta.fill_poly(
        canvas=np.zeros((200, 200)),
        vertices=[[[90., 90.], [110., 90.], [110., 110.], [90., 110.]]]
    )
    assert (canvas[80:120, 80:120] == 1).all()
    assert (canvas[:80] == 0).all()
    assert (canvas[:, :80] == 0).all()
    assert (canvas[121:] == 0).all()
    assert (canvas[:, 121:] == 0).all()

def test_fill_poly_array_single():
    rasta = Rasta(
        m_per_px=0.5,
        raster_fixpoint=(0.5, 0.5),
        world_fixpoint=(100, 100),
        camera_rotation=np.pi/2
    )
    canvas = rasta.fill_poly(
        canvas=np.zeros((200, 200)),
        vertices=[[90., 90.], [110., 90.], [110., 110.], [90., 110.]]
    )
    assert (canvas[80:120, 80:120] == 1).all()
    assert (canvas[:80] == 0).all()
    assert (canvas[:, :80] == 0).all()
    assert (canvas[121:] == 0).all()
    assert (canvas[:, 121:] == 0).all()

def test_fill_circle():
    rasta = Rasta(
        m_per_px=0.5,
        raster_fixpoint=(0.5, 0.5),
        world_fixpoint=(100, 100),
        camera_rotation=np.pi/2
    )
    rasta.fill_circle(
        np.zeros((2000, 2000)),
        (40.2, -23.572), 10,
    )
    rasta.fill_circle(
        np.zeros((2000, 2000)),
        [40.2, -23.572], np.array(10.1),
    )

def test_polylines():
    rasta = Rasta(
        m_per_px=0.5,
        raster_fixpoint=(0.5, 0.5),
        world_fixpoint=(0, 0),
        camera_rotation=np.pi/2
    )
    out = rasta.polylines(
        np.zeros((100, 100)),
        vertices=np.zeros((5, 10, 2))
    )
    assert out[50, 50] == 1

def test_polylines_single():
    rasta = Rasta(
        m_per_px=0.5,
        raster_fixpoint=(0.5, 0.5),
        world_fixpoint=(0, 0),
        camera_rotation=np.pi/2
    )
    out = rasta.polylines(
        np.zeros((100, 100)),
        vertices=np.zeros((10, 2))
    )
    assert out[50, 50] == 1

def test_polylines_iterable():
    rasta = Rasta(
        m_per_px=0.5,
        raster_fixpoint=(0.5, 0.5),
        world_fixpoint=(0, 0),
        camera_rotation=np.pi/2
    )
    out = rasta.polylines(
        np.zeros((100, 100)),
        vertices=[[[0, 0], [0, 0], [0, 0]]]
    )
    assert out[50, 50] == 1

def test_polylines_iterable_single():
    rasta = Rasta(
        m_per_px=0.5,
        raster_fixpoint=(0.5, 0.5),
        world_fixpoint=(0, 0),
        camera_rotation=np.pi/2
    )
    out = rasta.polylines(
        np.zeros((100, 100)),
        vertices=[[0, 0], [0, 0], [0, 0]]
    )
    assert out[50, 50] == 1
    assert out.sum() == 1

def test_rect_vertices():
    vertices = Rasta._rect_vertices(
        center=np.array([4, 3]),
        length=np.array([14]),
        width=np.array([10]),
        rotation=np.array([np.pi/2]),
    )
    expected = np.array([
        [4 - 10/2, 3 + 14/2],
        [4 - 10/2, 3 - 14/2],
        [4 + 10/2, 3 - 14/2],
        [4 + 10/2, 3 + 14/2],
    ])
    assert np.allclose(vertices, expected)

def test_rect_vertices_multiple():
    center = np.array([4, 3])
    length = np.array([14])
    width = np.array([10])
    rotation = np.array([np.pi/2])

    vertices = Rasta._rect_vertices(
        center=np.tile(center, (3, 1)),
        length=np.tile(length, (3, 1)),
        width=np.tile(width, (3, 1)),
        rotation=np.tile(rotation, (3, 1)),
    )

    vertices = np.array([
        [4 - 10/2, 3 + 14/2],
        [4 - 10/2, 3 - 14/2],
        [4 + 10/2, 3 - 14/2],
        [4 + 10/2, 3 + 14/2],
    ])
    expected = np.tile(vertices, (3, 1, 1))
    
    assert np.allclose(vertices, expected)

def test_rect_vertices_multiple_1dim():
    center = np.array([4, 3])
    length = np.array([14])
    width = np.array([10])
    rotation = np.array([np.pi/2])

    vertices = Rasta._rect_vertices(
        center=np.tile(center, (3,1)),
        length=np.tile(length, (3,)),
        width=np.tile(width, (3,)),
        rotation=np.tile(rotation, (3,)),
    )

    vertices = np.array([
        [4 - 10/2, 3 + 14/2],
        [4 - 10/2, 3 - 14/2],
        [4 + 10/2, 3 - 14/2],
        [4 + 10/2, 3 + 14/2],
    ])
    expected = np.tile(vertices, (3, 1, 1))
    
    assert np.allclose(vertices, expected)

def test_fill_tilted_rect():
    rasta = Rasta(
        m_per_px=0.5,
        raster_fixpoint=(0.5, 0.5),
        world_fixpoint=(100, 100),
        camera_rotation=np.pi/2
    )
    canvas = rasta.fill_tilted_rect(
        canvas=np.zeros((200, 200)),
        center=np.array([100, 100]),
        length=np.array([30]),
        width=np.array([20]),
        rotation=np.array([np.pi/2]),
    )
    assert (canvas[80:120, 70:130] == 1).all()
    assert (canvas[:80] == 0).all()
    assert (canvas[121:] == 0).all()
    assert (canvas[:, :70] == 0).all()
    assert (canvas[:, 131:] == 0).all()

def test_fill_tilted_rect_multiple():
    rasta = Rasta(
        m_per_px=0.5,
        raster_fixpoint=(0.5, 0.5),
        world_fixpoint=(100, 100),
        camera_rotation=np.pi/2
    )
    center = 100 * np.ones((10, 2))
    length = 30 * np.ones((10,))
    width = 20 * np.ones((10,))
    rotation = np.pi/2 * np.ones((10,))
    canvas = rasta.fill_tilted_rect(
        canvas=np.zeros((200, 200)),
        center=center,
        length=length,
        width=width,
        rotation=rotation,
    )
    assert (canvas[:80] == 0).all()
    assert (canvas[121:] == 0).all()
    assert (canvas[:, :70] == 0).all()
    assert (canvas[:, 131:] == 0).all()
    assert (canvas[80:120, 70:130] == 1).all()

def test_fill_tilted_rect_scalar():
    rasta = Rasta(
        m_per_px=0.5,
        raster_fixpoint=(0.5, 0.5),
        world_fixpoint=(100, 100),
        camera_rotation=np.pi/2
    )
    canvas = rasta.fill_tilted_rect(
        canvas=np.zeros((200, 200)),
        center=np.array([100, 100]),
        length=np.array(30),
        width=np.array(20),
        rotation=np.array([np.pi/2]),
    )
    assert (canvas[80:120, 70:130] == 1).all()
    assert (canvas[:80] == 0).all()
    assert (canvas[121:] == 0).all()
    assert (canvas[:, :70] == 0).all()
    assert (canvas[:, 131:] == 0).all()

def test_fill_tilted_rect_list():
    rasta = Rasta(
        m_per_px=0.5,
        raster_fixpoint=(0.5, 0.5),
        world_fixpoint=(100, 100),
        camera_rotation=np.pi/2
    )
    canvas = rasta.fill_tilted_rect(
        canvas=np.zeros((200, 200)),
        center=[100, 100],
        length=30,
        width=20,
        rotation=np.pi/2,
    )
    assert (canvas[80:120, 70:130] == 1).all()
    assert (canvas[:80] == 0).all()
    assert (canvas[121:] == 0).all()
    assert (canvas[:, :70] == 0).all()
    assert (canvas[:, 131:] == 0).all()

def test_fill_tilted_rect_list_multiple():
    rasta = Rasta(
        m_per_px=0.5,
        raster_fixpoint=(0.5, 0.5),
        world_fixpoint=(100, 100),
        camera_rotation=np.pi/2
    )
    canvas = rasta.fill_tilted_rect(
        canvas=np.zeros((200, 200)),
        center=[[100, 100], [100, 100]],
        length=[30, 30],
        width=[20, 20],
        rotation=[np.pi/2, np.pi/2],
    )
    assert (canvas[80:120, 70:130] == 1).all()
    assert (canvas[:80] == 0).all()
    assert (canvas[121:] == 0).all()
    assert (canvas[:, :70] == 0).all()
    assert (canvas[:, 131:] == 0).all()

def test_fill_tilted_rect_offset():
    rasta = Rasta(
        m_per_px=0.5,
        raster_fixpoint=(0.5, 0.5),
        world_fixpoint=(100, 100),
        camera_rotation=np.pi/2
    )
    canvas = rasta.fill_tilted_rect(
        canvas=np.zeros((200, 200)),
        center=[100, 90],
        length=30,
        width=20,
        rotation=np.pi/2,
    )
    assert (canvas[80:120, 50:110] == 1).all()
    assert (canvas[:80] == 0).all()
    assert (canvas[121:] == 0).all()
    assert (canvas[:, :50] == 0).all()
    assert (canvas[:, 111:] == 0).all()