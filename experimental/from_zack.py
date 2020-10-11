import numpy as np
from ase.io.extxyz import output_column_format
from ase.atoms import Atoms
from ase.calculators.calculator import all_properties, Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.spacegroup.spacegroup import Spacegroup
from ase.parallel import paropen
from ase.constraints import FixAtoms, FixCartesian
from ase.io.formats import index2range
from io import StringIO, UnsupportedOperation

per_atom_properties = ['forces',  'stresses', 'charges',  'magmoms', 'energies']
per_config_properties = ['energy', 'stress', 'dipole', 'magmom', 'free_energy']


def write_xyz(fileobj, images, comment='', columns=None, write_info=True,
              write_results=True, plain=False, vec_cell=False, append=False):
    """
    Write output in extended XYZ format

    Optionally, specify which columns (arrays) to include in output,
    whether to write the contents of the `atoms.info` dict to the
    XYZ comment line (default is True), the results of any
    calculator attached to this Atoms. The `plain` argument
    can be used to write a simple XYZ file with no additional information.
    `vec_cell` can be used to write the cell vectors as additional
    pseudo-atoms. If `append` is set to True, the file is for append (mode `a`),
    otherwise it is overwritten (mode `w`).

    See documentation for :func:`read_xyz()` for further details of the extended
    XYZ file format.
    """
    if isinstance(fileobj, str):
        mode = 'w'
        if append:
            mode = 'a'
        fileobj = paropen(fileobj, mode)

    if hasattr(images, 'get_positions'):
        images = [images]

    for atoms in images:
        natoms = len(atoms)

        if columns is None:
            fr_cols = None
        else:
            fr_cols = columns[:]

        if fr_cols is None:
            fr_cols = (['symbols', 'positions']
                       + [key for key in atoms.arrays.keys() if
                          key not in ['symbols', 'positions', 'numbers',
                                      'species', 'pos']])

        if vec_cell:
            plain = True

        if plain:
            fr_cols = ['symbols', 'positions']
            write_info = False
            write_results = False

        per_frame_results = {}
        per_atom_results = {}
        if write_results:
            calculator = atoms.calc
            if (calculator is not None
                    and isinstance(calculator, Calculator)):
                for key in all_properties:
                    value = calculator.results.get(key, None)
                    if value is None:
                        # skip missing calculator results
                        continue
                    if (key in per_atom_properties and len(value.shape) >= 1
                        and value.shape[0] == len(atoms)):
                        # per-atom quantities (forces, energies, stresses)
                        per_atom_results[key] = value
                    elif key in per_config_properties:
                        # per-frame quantities (energy, stress)
                        # special case for stress, which should be converted
                        # to 3x3 matrices before writing
                        if key == 'stress':
                            xx, yy, zz, yz, xz, xy = value
                            value = np.array(
                                [(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])
                        per_frame_results[key] = value

        # Move symbols and positions to first two properties
        if 'symbols' in fr_cols:
            i = fr_cols.index('symbols')
            fr_cols[0], fr_cols[i] = fr_cols[i], fr_cols[0]

        if 'positions' in fr_cols:
            i = fr_cols.index('positions')
            fr_cols[1], fr_cols[i] = fr_cols[i], fr_cols[1]

        # Check first column "looks like" atomic symbols
        if fr_cols[0] in atoms.arrays:
            symbols = atoms.arrays[fr_cols[0]]
        else:
            symbols = atoms.get_chemical_symbols()

        if natoms > 0 and not isinstance(symbols[0], str):
            raise ValueError('First column must be symbols-like')

        # Check second column "looks like" atomic positions
        pos = atoms.arrays[fr_cols[1]]
        if pos.shape != (natoms, 3) or pos.dtype.kind != 'f':
            raise ValueError('Second column must be position-like')

        # if vec_cell add cell information as pseudo-atoms
        if vec_cell:
            pbc = list(atoms.get_pbc())
            cell = atoms.get_cell()

            if True in pbc:
                nPBC = 0
                for i, b in enumerate(pbc):
                    if b:
                        nPBC += 1
                        symbols.append('VEC' + str(nPBC))
                        pos = np.vstack((pos, cell[i]))
                # add to natoms
                natoms += nPBC
                if pos.shape != (natoms, 3) or pos.dtype.kind != 'f':
                    raise ValueError(
                        'Pseudo Atoms containing cell have bad coords')

        # Move mask
        if 'move_mask' in fr_cols:
            cnstr = atoms._get_constraints() # zulissi: SINGLE LINE CHANGE! 
            if len(cnstr) > 0:
                c0 = cnstr[0]
                if isinstance(c0, FixAtoms):
                    cnstr = np.ones((natoms,), dtype=np.bool)
                    for idx in c0.index:
                        cnstr[idx] = False
                elif isinstance(c0, FixCartesian):
                    for i in range(len(cnstr)):
                        idx = cnstr[i].a
                        cnstr[idx] = cnstr[i].mask
                    cnstr = np.asarray(cnstr)
            else:
                fr_cols.remove('move_mask')

        # Collect data to be written out
        arrays = {}
        for column in fr_cols:
            if column == 'positions':
                arrays[column] = pos
            elif column in atoms.arrays:
                arrays[column] = atoms.arrays[column]
            elif column == 'symbols':
                arrays[column] = np.array(symbols)
            elif column == 'move_mask':
                arrays[column] = cnstr
            else:
                raise ValueError('Missing array "%s"' % column)

        if write_results:
            for key in per_atom_results:
                if key not in fr_cols:
                    fr_cols += [key]
                else:
                    warnings.warn('write_xyz() overwriting array "{0}" present '
                                  'in atoms.arrays with stored results '
                                  'from calculator'.format(key))
            arrays.update(per_atom_results)

        comm, ncols, dtype, fmt = output_column_format(atoms,
                                                       fr_cols,
                                                       arrays,
                                                       write_info,
                                                       per_frame_results)

        if plain or comment != '':
            # override key/value pairs with user-speficied comment string
            comm = comment.rstrip()
            if '\n' in comm:
                raise ValueError('Comment line should not have line breaks.')

        # Pack fr_cols into record array
        data = np.zeros(natoms, dtype)
        for column, ncol in zip(fr_cols, ncols):
            value = arrays[column]
            if ncol == 1:
                data[column] = np.squeeze(value)
            else:
                for c in range(ncol):
                    data[column + str(c)] = value[:, c]

        nat = natoms
        if vec_cell:
            nat -= nPBC
        # Write the output
        fileobj.write('%d\n' % nat)
        fileobj.write('%s\n' % comm)
        for i in range(natoms):
            fileobj.write(fmt % tuple(data[i]))
