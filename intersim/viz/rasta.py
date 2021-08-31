import numpy as np
import cv2

class Rasta:

    def __init__(self, m_per_px, raster_fixpoint, world_fixpoint, camera_rotation):
        self._m_per_px = m_per_px
        self._raster_fixpoint = np.array(raster_fixpoint)
        self._world_fixpoint = np.array(world_fixpoint)
        self._camera_rotation = camera_rotation

    @staticmethod
    def _rotation_matrix2(r):
        return np.array([
            [np.cos(r), -np.sin(r)],
            [np.sin(r), np.cos(r)],
        ])

    @staticmethod
    def _isotropic_scale_matrix2(s):
        return np.diag([s, s])

    def _world_to_raster(self, raster_shape, points):
        """Transform world coordinates (x right, y up) to cv2 image coordinates (x right, y down).
        
        Args:
            raster_shape: (height, width)
            points: np.ndarray([..., 2])
        
        Returns:
            A np.ndarray of coordinates, centered and rotated around the world and raster fixpoint.
        """
        points_list = np.reshape(points, (-1, 2))
        raster_points_list = points_list.copy()
        raster_points_list -= self._world_fixpoint[np.newaxis] # -= runs in-place in this case
        raster_points_list = raster_points_list @ self._rotation_matrix2(-self._camera_rotation).T
        raster_points_list = raster_points_list @ self._isotropic_scale_matrix2(1 / self._m_per_px).T
        raster_points_list *= np.array([[1, -1]]) # flip y axis
        raster_points_list += (self._raster_fixpoint * np.array(raster_shape))[np.newaxis]

        raster_points = np.reshape(raster_points_list, points.shape)
        return np.array(raster_points, dtype=int)

    def fill_poly(self, canvas, vertices, value=1.0):
        """Fills the area bounded by one or more polygons with `value`.
        
        Args:
            canvas: np.ndarray([h, w]) **Warning: this gets modified in-place!**
            vertices: np.ndarray([n_poly, n_vertex, 2]) or np.ndarray([n_vertex, 2]) or Array[n_poly, n_vertex, 2] or Array[n_vertex, 2]
            color: fill value
        Returns:
            The updated canvas.
        """
        vertices = np.array(vertices)
        if vertices.ndim < 3:
            vertices = vertices[np.newaxis]
        points = self._world_to_raster(canvas.shape, vertices)
        for p in points:
            cv2.fillPoly(canvas, p[np.newaxis], value)
        return canvas

    def fill_circle(self, canvas, center, radius=0, color=1.0):
        """Fills a single circle with given center and radius.
        
        Args:
            canvas: np.ndarray([h, w]) **Warning: this gets modified in-place!**
            center: circle center (2-dimensional np.ndarray or Iterable)
            radius: circle radius
            color: fill value
        
        Returns:
            The updated canvas.
        """
        center = np.array(center)
        center = self._world_to_raster(canvas.shape, center)
        cv2.circle(canvas, center, int(radius / self._m_per_px), color, thickness=-1) # filled
        return canvas

    def polylines(self, canvas, vertices, color=1.0, thickness=1):
        """Draws several polygonal curves.
        
        Args:
            canvas: np.ndarray([h, w]) **Warning: this gets modified in-place!**
            vertices: np.ndarray([(n_lines, )n_vertices, 2]) or equivalent Iterable
            color: edge color
            thickness: edge thickness
        Returns:
            The updated canvas.
        """
        vertices = np.array(vertices)
        if vertices.ndim < 3:
            vertices = vertices[np.newaxis]
        coords = self._world_to_raster(canvas.shape, vertices)
        cv2.polylines(canvas, coords, isClosed=False, color=color, thickness=thickness)
        return canvas

    @staticmethod
    def _rect_vertices(center, length, width, rotation):
        front = length[..., np.newaxis] / 2 * np.stack([np.cos(rotation), np.sin(rotation)], axis=-1)
        left = width[..., np.newaxis] / 2 * np.stack([-np.sin(rotation), np.cos(rotation)], axis=-1)
        p0 = center + front + left
        p1 = center - front + left
        p2 = center - front - left
        p3 = center + front - left
        coords = np.stack([p0, p1, p2, p3], axis=-2)
        return coords

    def fill_tilted_rect(self, canvas, center, length, width, rotation, color=1.0):
        """Fills the area bounded by one or more tilted rectangles with `value`.
        
        Args:
            canvas: np.ndarray([h, w]) **Warning: this gets modified in-place!**
            center: rectangle centers (x, y)
            length: longitudinal rectangle sizes
            width: lateral rectangle sizes
            rotation: rectangle rotation angles
        
        Returns:
            The updated canvas.
        """
        center = np.array(center)
        length = np.array(length)
        width = np.array(width)
        rotation = np.array(rotation)
        vertices = self._rect_vertices(center, length, width, rotation)
        self.fill_poly(canvas, vertices, color)
        return canvas