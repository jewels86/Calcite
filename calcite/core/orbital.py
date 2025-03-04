from numba import int64, types, typed, typeof
from numba.experimental import jitclass
from calcite.core.particle import particle_type, Particle

orbital_spec = [
    ('n', int64),
    ('l', int64),
    ('m', int64),
    ('electrons', types.ListType(particle_type)),
    ('debug_mode', types.boolean)
]

@jitclass(orbital_spec)
class Orbital:
    """
    A class representing an atomic orbital.

    Attributes:
    - n (int): the principal quantum number
    - l (int): the azimuthal quantum number
    - m (int): the magnetic quantum number
    - electrons (list): a list of electrons in the orbital
    - debug_mode (bool): flag for debug mode
    """
    def __init__(self, n, l, m, electrons):
        """
        Initializes an Orbital object with the given quantum numbers and electrons.

        Args:
        - n (int): the principal quantum number
        - l (int): the azimuthal quantum number
        - m (int): the magnetic quantum number
        - electrons (list): a list of electrons in the orbital
        """
        self.n = n # principal quantum number
        self.l = l # azimuthal quantum number
        self.m = m # magnetic quantum number
        self.electrons = electrons

    def can_add(self, electron: Particle):
        """
        Checks if an electron can be added to the orbital.

        Args:
        - electron (Particle): the electron to be added

        Returns:
        - bool: True if the electron can be added, False otherwise
        """
        if self.debug_mode: print(f"Orbital.can_add: Checking if electron with spin ({'up' if electron.spin == 0.5 else 'down'}) can be added to orbital {self.n}, {self.l}, {self.m}")
        if self.debug_mode: print(f"Orbital.can_add: Current electrons in orbital: {len(self.electrons)} ({('up' if self.electrons[0].spin == 0.5 else 'down') if len(self.electrons) > 0 else 'None'\
                                                                                                           }, {('up' if self.electrons[1].spin == 0.5 else 'down') if len(self.electrons) > 1 else 'None'})")
        can_add = len(self.electrons) < 2 and (len(self.electrons) == 0 or electron.spin != self.electrons[0].spin)
        if self.debug_mode: print(f"Orbital.can_add: Can add: {can_add}")
        return can_add
        # orbitals can only hold two electrons - one with spin up and one with spin down

    def add(self, electron: Particle):
        """
        Adds an electron to the orbital if possible.

        Args:
        - electron (Particle): the electron to be added

        Returns:
        - bool: True if the electron was added, False otherwise
        """
        if self.can_add(electron): 
            self.electrons.append(electron) 
            return True
        return False
        

orbital_type = typeof(Orbital(1, 0, 0, typed.List.empty_list(particle_type)))