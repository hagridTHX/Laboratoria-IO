# zad4.py
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [12, 8]

EPSILON = 0.0001


def reflect(vector, normal_vector):
    n_dot_l = np.dot(vector, normal_vector)
    return vector - normal_vector * (2 * n_dot_l)


# zalamanie promienia wg prawa Snella, None gdy calkowite wewnetrzne odbicie
def refract(vector, normal_vector, eta_ratio):
    cos_theta = min(-np.dot(vector, normal_vector), 1.0)
    r_out_perp = eta_ratio * (vector + cos_theta * normal_vector)
    k = 1.0 - np.dot(r_out_perp, r_out_perp)
    if k < 0:
        return None
    r_out_parallel = -np.sqrt(k) * normal_vector
    return r_out_perp + r_out_parallel


def normalize(vector):
    return vector / np.sqrt((vector**2).sum())


class Ray:
    def __init__(self, starting_point, direction):
        self.starting_point = starting_point
        self.direction = direction


class Light:
    def __init__(self, position):
        self.position = position
        self.ambient = np.array([0, 0, 0])
        self.diffuse = np.array([0, 1, 1])
        self.specular = np.array([1, 1, 0])


class SceneObject:
    def __init__(
        self,
        ambient=np.array([0, 0, 0]),
        diffuse=np.array([0.6, 0.7, 0.8]),
        specular=np.array([0.8, 0.8, 0.8]),
        shining=25,
        transparency=0.0,
        ior=1.5,
    ):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shining = shining
        self.transparency = transparency
        self.ior = ior

    def get_normal(self, cross_point):
        raise NotImplementedError

    def trace(self, ray):
        raise NotImplementedError

    def get_color(self, cross_point, obs_vector, scene):
        color = self.ambient * scene.ambient
        light = scene.light

        normal = self.get_normal(cross_point)
        light_vector = normalize(light.position - cross_point)
        n_dot_l = np.dot(light_vector, normal)
        reflection_vector = normalize(reflect(-1 * light_vector, normal))

        v_dot_r = np.dot(reflection_vector, -obs_vector)
        if v_dot_r < 0:
            v_dot_r = 0

        if n_dot_l > 0:
            color += (
                (self.diffuse * light.diffuse * n_dot_l)
                + (self.specular * light.specular * v_dot_r**self.shining)
                + (self.ambient * light.ambient)
            )

        return color


class Sphere(SceneObject):
    def __init__(
        self,
        position,
        radius,
        ambient=np.array([0, 0, 0]),
        diffuse=np.array([0.6, 0.7, 0.8]),
        specular=np.array([0.8, 0.8, 0.8]),
        shining=25,
        transparency=0.0,
        ior=1.5,
    ):
        super(Sphere, self).__init__(
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            shining=shining,
            transparency=transparency,
            ior=ior,
        )
        self.position = position
        self.radius = radius

    def get_normal(self, cross_point):
        return normalize(cross_point - self.position)

    def trace(self, ray):
        distance = ray.starting_point - self.position

        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(ray.direction, distance)
        c = np.dot(distance, distance) - self.radius**2
        d = b**2 - 4 * a * c

        if d < 0:
            return (None, None)

        sqrt_d = d**0.5
        denominator = 1 / (2 * a)

        if d > 0:
            r1 = (-b - sqrt_d) * denominator
            r2 = (-b + sqrt_d) * denominator
            if r1 < EPSILON:
                if r2 < EPSILON:
                    return (None, None)
                r1 = r2
        else:
            r1 = -b * denominator
            if r1 < EPSILON:
                return (None, None)

        cross_point = ray.starting_point + r1 * ray.direction
        return cross_point, r1


class Camera:
    def __init__(self, position=np.array([0, 0, -3]), look_at=np.array([0, 0, 0])):
        self.z_near = 1
        self.pixel_height = 500
        self.pixel_width = 700
        self.povy = 45
        look = normalize(look_at - position)
        self.up = normalize(np.cross(np.cross(look, np.array([0, 1, 0])), look))
        self.position = position
        self.look_at = look_at
        self.direction = normalize(look_at - position)
        aspect = self.pixel_width / self.pixel_height
        povy = self.povy * np.pi / 180
        self.world_height = 2 * np.tan(povy / 2) * self.z_near
        self.world_width = aspect * self.world_height

        center = self.position + self.direction * self.z_near
        width_vector = normalize(np.cross(self.up, self.direction))
        self.translation_vector_x = width_vector * -(self.world_width / self.pixel_width)
        self.translation_vector_y = self.up * -(self.world_height / self.pixel_height)
        self.starting_point = (
            center
            + width_vector * (self.world_width / 2)
            + (self.up * self.world_height / 2)
        )

    def get_world_pixel(self, x, y):
        return self.starting_point + self.translation_vector_x * x + self.translation_vector_y * y


class Scene:
    def __init__(self, objects, light, camera):
        self.objects = objects
        self.light = light
        self.camera = camera
        self.ambient = np.array([0.1, 0.1, 0.1])
        self.background = np.array([0, 0, 0])


class RayTracer:
    def __init__(self, scene):
        self.scene = scene

    def generate_image(self):
        camera = self.scene.camera
        image = np.zeros((camera.pixel_height, camera.pixel_width, 3))
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                world_pixel = camera.get_world_pixel(x, y)
                direction = normalize(world_pixel - camera.position)
                image[y][x] = self._get_pixel_color(Ray(world_pixel, direction))
        return image

    def _get_pixel_color(self, ray, depth=3):
        obj, distance, cross_point = self._get_closest_object(ray)

        if not obj:
            return self.scene.background

        return obj.get_color(cross_point, ray.direction, self.scene)

    def _get_closest_object(self, ray):
        closest = None
        min_distance = np.inf
        min_cross_point = None
        for obj in self.scene.objects:
            cross_point, distance = obj.trace(ray)
            if cross_point is not None and distance < min_distance:
                min_distance = distance
                closest = obj
                min_cross_point = cross_point

        return (closest, min_distance, min_cross_point)


# tracer z odbiciem i przezroczystoscia
class MyRayTracer2(RayTracer):
    def _get_pixel_color(self, ray, depth=3):
        if depth == 0:
            return self.scene.background

        obj, distance, cross_point = self._get_closest_object(ray)
        if not obj:
            return self.scene.background

        local_color = obj.get_color(cross_point, ray.direction, self.scene)

        # sprawdzam czy wchodzimy do obiektu czy z niego wychodzimy
        unit_dir = normalize(ray.direction)
        normal = obj.get_normal(cross_point)
        front_face = np.dot(unit_dir, normal) < 0
        if not front_face:
            normal = -normal

        # promien odbity
        reflection_vector = normalize(reflect(unit_dir, normal))
        reflection_ray = Ray(cross_point + reflection_vector * EPSILON, reflection_vector)
        reflected_color = self._get_pixel_color(reflection_ray, depth - 1)

        # promien zalamany, stosunek wspolczynnikow zalezy od kierunku
        eta_ratio = (1.0 / obj.ior) if front_face else (obj.ior / 1.0)
        refracted_dir = refract(unit_dir, normal, eta_ratio)

        refracted_color = np.array([0.0, 0.0, 0.0])
        if refracted_dir is not None:
            refracted_ray = Ray(cross_point + refracted_dir * EPSILON, refracted_dir)
            refracted_color = self._get_pixel_color(refracted_ray, depth - 1)

        # waga zalamania zalezy od przezroczystosci, reszta idzie na lokalny i odbity
        base_local = 0.4
        base_reflect = 0.3
        base_refract = 0.3

        w_refract = base_refract * obj.transparency
        rem = 1.0 - w_refract
        w_local = rem * (base_local / (base_local + base_reflect))
        w_reflect = rem * (base_reflect / (base_local + base_reflect))

        return w_local * local_color + w_reflect * reflected_color + w_refract * refracted_color


scene_transparent = Scene(
    objects=[
        Sphere(position=np.array([0, 0, 0]), radius=1.3, diffuse=np.array([0.2, 0.7, 1.0]), transparency=0.6, ior=1.4),
        Sphere(position=np.array([2.2, -0.3, 1.5]), radius=0.8, diffuse=np.array([1, 0, 1]), transparency=0.2, ior=1.5),
        Sphere(position=np.array([-2.4, 0.2, 1.0]), radius=0.9, diffuse=np.array([0, 1, 0])),
    ],
    light=Light(position=np.array([3, 6, 8])),
    camera=Camera(position=np.array([0, 0, 8])),
)

rt_transparent = MyRayTracer2(scene_transparent)
image_transparent = np.clip(rt_transparent.generate_image(), 0, 1)
plt.imshow(image_transparent)
plt.axis("off")
plt.show()