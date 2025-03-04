import numpy as np
from calcite.core import Particle, electron, proton, neutron, \
    particle_type, composite_particle_type, orbital_type, Orbital
from calcite.constants import NAN_VECTOR
from calcite.core.exceptions import UnassignedElectronsError
from numba import njit, float64, int64, types, typed, typeof
from numba.experimental import jitclass
from numba.types import Tuple

orbital_key_type = Tuple((int64, int64, int64))
awi_electron = types.Tuple([int64, particle_type])
awi_electron_electron = types.Tuple([int64, particle_type, particle_type])
atom_spec = [
    ('position', float64[:]), # position vector (x, y, z)
    ('velocity', float64[:]), # velocity vector (vx, vy, vz)
    ('protons', types.ListType(composite_particle_type)), # list of protons
    ('neutrons', types.ListType(composite_particle_type)), # list of neutrons
    ('electrons', types.ListType(particle_type)), # list of electrons
    ('orbitals', types.DictType(orbital_key_type, int64)), # dictionary of orbital parameters -> index
    ('_orbitals', types.ListType(orbital_type)), # list of orbitals
    ('ionic_bonds', types.ListType(types.Tuple([int64, particle_type]))), # list of ionic bonds (atom world index -> electron)
    ('covalent_bonds', types.ListType(awi_electron_electron)), # list of covalent bonds (atom world index -> electron)
    ('unassigned', types.ListType(particle_type)), # list of unassigned electrons
    ('index', int64), # atom world index
    ('debug_mode', types.boolean)
]

@jitclass(atom_spec)
class Atom:
    """
    A class representing an atom.

    Attributes:
    - position (np.ndarray): the position of the atom in 3D space
    - velocity (np.ndarray): the velocity of the atom in 3D space
    - protons (list): a list of protons in the atom
    - neutrons (list): a list of neutrons in the atom
    - electrons (list): a list of electrons in the atom
    - orbitals (dict): a dictionary of orbital parameters to index
    - _orbitals (list): a list of orbitals
    - ionic_bonds (list): a list of ionic bonds
    - covalent_bonds (list): a list of covalent bonds
    - unassigned (list): a list of unassigned electrons
    - index (int): the atom world index
    - debug_mode (bool): flag for debug mode
    """
    def __init__(self, position, velocity, n_protons, n_neutrons, n_electrons, debug):
        """
        Initializes an Atom object with the given position, velocity, number of protons, neutrons, electrons, and debug mode.

        Args:
        - position (list[float]): the position of the atom in 3D space
        - velocity (list[float]): the velocity of the atom in 3D space
        - n_protons (int): the number of protons in the atom
        - n_neutrons (int): the number of neutrons in the atom
        - n_electrons (int): the number of electrons in the atom
        - debug (bool): flag for debug mode
        """
        self.position = position
        self.velocity = velocity
        self.protons = typed.List([proton() for _ in range(n_protons)])
        self.neutrons = typed.List([neutron() for _ in range(n_neutrons)])
        self.electrons = typed.List.empty_list(particle_type)
        self.orbitals = typed.Dict.empty(orbital_key_type, int64)
        self._orbitals = typed.List.empty_list(orbital_type)
        self.ionic_bonds = typed.List.empty_list(awi_electron)
        self.covalent_bonds = typed.List.empty_list(awi_electron_electron)
        self.unassigned = typed.List.empty_list(particle_type)
        self.index = -1
        self.debug_mode = debug
        self.setup_electrons(n_electrons)
        self.configure()

    def setup_electrons(self, n_electrons):
        """
        Sets up the electrons in the atom.

        Args:
        - n_electrons (int): the number of electrons to be set up
        """
        added = 0
        orbitals_order = [(n, l) for n in range(1, 7) for l in range(n)]
        
        for n, l in orbitals_order:
            for m in range(-l, l+1):
                if added < n_electrons:
                    new_electron = electron(n, l, m)
                    new_electron.spin = 0.5 if added % 2 == 0 else -0.5
                    self.electrons.append(new_electron)
                    if self.debug_mode: print(f"Added electron with spin {'up' if new_electron.spin == 0.5 else 'down'} to orbital {n}, {l}, {m}")
                    added += 1
        
    def configure(self, order: list = None) -> None:
        """
        Configures the orbitals and assigns electrons to them.

        Args:
        - order (list, optional): the order of orbitals. Defaults to None.
        """
        added = 0
        if order is None: order = [(n, l) for n in range(1, 7) for l in range(n)]
        capacity = sum([2 * (2 * l + 1) for _, l in order])
        unassigned = typed.List.empty_list(particle_type)

        for orbital in self._orbitals:
            orbital.electrons.clear()

        for n, l in order:
            for m in range(-l, l+1):
                if (n, l, m) not in self.orbitals:
                    self.orbitals[(n, l, m)] = len(self._orbitals)
                    self._orbitals.append(Orbital(n, l, m, typed.List.empty_list(particle_type)))
                orbital = self._orbitals[self.orbitals[(n, l, m)]]
                for _ in range(2):
                    if added < len(self.electrons):
                        electron = self.electrons[added]
                        if orbital.add(electron):
                            if self.debug_mode: 
                                print(f"Assigned electron with spin {'up' if electron.spin == 0.5 else 'down'} to orbital {n}, {l}, {m}")
                            added += 1
                        else:
                            unassigned.append(electron)
                    else:
                        break

        max_n = max([orbital.n for orbital in self._orbitals])
        for n in range(1, max_n + 2):
            for l in range(n):
                for m in range(-l, l + 1):
                    if (n, l, m) not in self.orbitals:
                        self.orbitals[(n, l, m)] = len(self._orbitals)
                        self._orbitals.append(Orbital(n, l, m, typed.List.empty_list(particle_type)))

        to_stay = [orbital for orbital in self._orbitals if len(orbital.electrons) != 0]
        to_remove = [orbital for orbital in self._orbitals if len(orbital.electrons) == 0]
        for orbital in to_remove: self.orbitals.pop((orbital.n, orbital.l, orbital.m), None)
        self._orbitals = typed.List(to_stay)

        if added < len(self.electrons):
            if self.debug_mode:
                print(f"Atom.configure: {len(self.electrons) - added} electrons remain unassigned.")
                print(f"Unassigned electrons will be stored in the unassigned list.")
            raise UnassignedElectronsError("Unassigned electrons remain.")
        
        if len(self.electrons) > capacity:
            if self.debug_mode:
                print(f"Atom.configure: Atom has {len(self.electrons)} electrons, but only {capacity} can be assigned.")
                print(f"Excess electrons will be stored in the unassigned list.")
            raise UnassignedElectronsError("Excess electrons remain.")
        self.unassigned = unassigned
    
    def add_electron(self, electron: Particle) -> bool:
        """
        Adds an electron to the atom.

        Args:
        - electron (Particle): the electron to be added

        Returns:
        - bool: True if the electron was added, False otherwise
        """
        for orbital in self._orbitals:
            if orbital.can_add(electron):
                self.electrons.append(electron)
                self.configure()
                return True
        return False
    
    def remove_electron(self) -> bool:
        """
        Removes an electron from the atom.

        Returns:
        - bool: True if an electron was removed, False otherwise
        """
        if len(self.electrons) > 0:
            self.electrons.pop()
            self.configure()
            return True
        return False
    
    def remove_electron(self, electron: Particle) -> bool:
        """
        Removes a specific electron from the atom.

        Args:
        - electron (Particle): the electron to be removed

        Returns:
        - bool: True if the electron was removed, False otherwise
        """
        if electron.index == -1:
            return False
        new_electrons = typed.List.empty_list(particle_type)
        removed = False

        for e in self.electrons:
            if e.index != electron.index or removed:
                new_electrons.append(e)
            else:
                removed = True

        if removed:
            self.electrons = new_electrons
            self.configure()
            return True
        return False
    
    def covalent_bond(self, other: "Atom") -> bool:
        """
        Forms a covalent bond with another atom.

        Args:
        - other (Atom): the other atom to form a bond with

        Returns:
        - bool: True if the bond was formed, False otherwise
        """
        if self.stable or other.stable:
            return False
        
        valence_self = self.valence_electrons
        valence_other = other.valence_electrons

        unpaired_self = [e for e in valence_self if e.spin == 0.5]
        unpaired_other = [e for e in valence_other if e.spin == 0.5]

        if len(unpaired_self) == 0 or len(unpaired_other) == 0:
            return False

        original_self_spin = unpaired_self[0].spin
        original_other_spin = unpaired_other[0].spin
        unpaired_self[0].spin = -0.5
        unpaired_other[0].spin = 0.5

        if not self.add_electron_to_valence_shell(unpaired_other[0]) or \
            not other.add_electron_to_valence_shell(unpaired_self[0]):
            unpaired_self[0].spin = 0.5
            unpaired_other[0].spin = -0.5
            if not self.add_electron_to_valence_shell(unpaired_other[0]) or \
                not other.add_electron_to_valence_shell(unpaired_self[0]):
                unpaired_self[0].spin = original_self_spin
                unpaired_other[0].spin = original_other_spin
                return False
        
        self.covalent_bonds.append((other.index, unpaired_self[0], unpaired_other[0]))
        other.covalent_bonds.append((self.index, unpaired_other[0], unpaired_self[0]))

        return True

    def add_electron_to_valence_shell(self, electron: Particle) -> bool:
        """
        Adds an electron to the valence shell of the atom.

        Args:
        - electron (Particle): the electron to be added

        Returns:
        - bool: True if the electron was added, False otherwise
        """
        valence_shell = max([orbital.n for orbital in self._orbitals])
        if self.debug_mode: print(f"Adding electron to valence shell {valence_shell}")
        
        for orbital in self._orbitals:
            if orbital.n == valence_shell and orbital.can_add(electron):
                if self.debug_mode: print(f"Adding electron to orbital {orbital.n}, {orbital.l}, {orbital.m}")
                orbital.add(electron)
                self.electrons.append(electron)
                return True
        
        for l in range(valence_shell):
            for m in range(-l, l + 1):
                if (valence_shell, l, m) not in self.orbitals:
                    if self.debug_mode: print(f"Creating new orbital {valence_shell}, {l}, {m} for electron")
                    self.orbitals[(valence_shell, l, m)] = len(self._orbitals)
                    new_orbital = Orbital(valence_shell, l, m, typed.List.empty_list(particle_type))
                    self._orbitals.append(new_orbital)
                    if new_orbital.can_add(electron):
                        if self.debug_mode: print(f"Adding electron to new orbital {valence_shell}, {l}, {m}")
                        new_orbital.add(electron)
                        self.electrons.append(electron)
                        return True
        
        if self.debug_mode: print("Failed to add electron to valence shell")
        return False

    def ionic_bond(self, other: "Atom") -> bool:
        """
        Forms an ionic bond with another atom.

        Args:
        - other (Atom): the other atom to form a bond with

        Returns:
        - bool: True if the bond was formed, False otherwise
        """
        if self.stable or other.stable:
            return False
        
        unpaired_self = [e for e in self.valence_electrons if e.spin == 0.5]
        unpaired_other = [e for e in other.valence_electrons if e.spin == 0.5]

        if len(unpaired_self) > 0 and len(unpaired_other) > 0:
            e_to_transfer = unpaired_self[0]
            self.electrons.remove(e_to_transfer)
            other.electrons.append(e_to_transfer)
            e_to_transfer.spin = -0.5
            self.ionic_bonds.append((other.index, e_to_transfer))
            other.ionic_bonds.append((self.index, e_to_transfer))

            return True
        return False

    @property
    def valence_electrons(self):
        """
        Returns the valence electrons of the atom.

        Returns:
        - list: a list of valence electrons
        """
        valence_electrons = []

        for orbital in self._orbitals:
            if orbital.n == max([o.n for o in self._orbitals]):
                valence_electrons.extend(orbital.electrons)

        return valence_electrons

    @property
    def stable(self):
        """
        Checks if the atom is stable.

        Returns:
        - bool: True if the atom is stable, False otherwise
        """
        valence_electrons = 0
        max_valence = 8 if self.atomic_number < 18 else 18

        for orbital in self._orbitals:
            if orbital.n == max([o.n for o in self._orbitals]):
                valence_electrons += len(orbital.electrons)
        
        return valence_electrons == max_valence or valence_electrons == 0 or valence_electrons == 2

    @property
    def mass(self):
        """
        Returns the mass of the atom.

        Returns:
        - float: the mass of the atom
        """
        return sum([proton.mass for proton in self.protons]) \
            + sum([neutron.mass for neutron in self.neutrons]) \
            + sum([_electron.mass for _electron in self.electrons])
    
    @property
    def atomic_number(self):
        """
        Returns the atomic number of the atom.

        Returns:
        - int: the atomic number of the atom
        """
        return len(self.protons)
    
    @property
    def charge(self):
        """
        Returns the charge of the atom.

        Returns:
        - float: the charge of the atom
        """
        return sum([proton.charge for proton in self.protons]) \
            - sum([_electron.charge for _electron in self.electrons])
    
    @property
    def spin(self):
        """
        Returns the spin of the atom.

        Returns:
        - float: the spin of the atom
        """
        return sum([proton.spin for proton in self.protons]) \
            + sum([neutron.spin for neutron in self.neutrons]) \
            + sum([_electron.spin for _electron in self.electrons])
    
    @property
    def momentum(self):
        """
        Returns the momentum of the atom.

        Returns:
        - float: the momentum of the atom
        """
        return np.linalg.norm(self.mass * self.velocity)

    def _debug_mode(self):
        """
        Enables debug mode for the atom and its orbitals.
        """
        self.debug_mode = True
        for orbital in self._orbitals:
            orbital.debug_mode = True

@njit
def atom(n_protons: int, n_neutrons: int, n_electrons: int, position: list[float] | None = None, 
         velocity: list[float] | None = None, debug: bool = False) -> Atom:
    """
    Creates a new atom with the given number of protons, neutrons, electrons, position, velocity, and debug mode.

    Args:
        n_protons (int): The number of protons in the atom.
        n_neutrons (int): The number of neutrons in the atom.
        n_electrons (int): The number of electrons in the atom.
        position (list[float] | None, optional): The position of the atom. Defaults to None.
        velocity (list[float] | None, optional): The velocity of the atom. Defaults to None.
        debug (bool, optional): Flag for debug mode. Defaults to False.

    Returns:
        Atom: A new atom object
    """
    if position is None:
        position = NAN_VECTOR
    if velocity is None:
        velocity = NAN_VECTOR
    return Atom(position, velocity, n_protons, n_neutrons, n_electrons, debug)

atom_type = typeof(atom(0, 0, 0))