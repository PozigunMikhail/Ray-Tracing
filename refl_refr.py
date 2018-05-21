import numpy as np
import math
import enum


class MaterialTypes(enum.Enum):
    DIFFUSE = 0
    REFLECTION = 1
    REFLECTION_AND_REFRACTION = 2


def get_fresnel_coeffs(incoming_ray, normal, refr_index2, refr_index1=1.0):
    cosi = np.clip(np.dot(incoming_ray, normal), 0.0, 1.0)
    etai = refr_index1
    etat = refr_index2
    n = np.copy(normal)
    if cosi > 0.0:
        etai, etat = etat, etai
    sint = etai / etat * math.sqrt(max(0.0, 1 - cosi ** 2))
    if sint >= 1.0:
        return 1.0, 0.0
    cost = max(0.0, 1 - sint ** 2)
    cosi = abs(cosi)
    R_orth = (etat * cosi - etai * cost) / (etat * cosi + etai * cost)
    R_par = (etai * cosi - etat * cost) / (etai * cosi + etat * cost)
    kr = (R_orth ** 2 + R_par ** 2) / 2.0
    return kr, 1.0 - kr


def get_reflected_ray(incoming_ray, normal):
    # incoming_ray_n = incoming_ray / np.linalg.norm(incoming_ray)
    reflected_ray = incoming_ray - 2 * np.dot(incoming_ray, normal) * normal
    return reflected_ray


def get_refracted_ray(incoming_ray, normal, refr_index2, refr_index1=1.0):
    # incoming_ray_n = incoming_ray / np.linalg.norm(incoming_ray)
    cosi = np.clip(np.dot(incoming_ray, normal), 0.0, 1.0)
    etai = refr_index1
    etat = refr_index2
    n = np.copy(normal)
    if cosi < 0.0:
        cosi = -cosi
    else:
        etai, etat = etat, etai
        n = -n
    eta = etai / etat
    k = 1 - (eta ** 2) * (1 - cosi ** 2)
    if k < 0.0:
        return None
    else:
        return eta * incoming_ray + (eta * cosi - math.sqrt(k)) * n
