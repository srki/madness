/*
 This file is part of MADNESS.

 Copyright (C) 2007,2010 Oak Ridge National Laboratory

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

 For more information please contact:

 Robert J. Harrison
 Oak Ridge National Laboratory
 One Bethel Valley Road
 P.O. Box 2008, MS-6367

 email: harrisonrj@ornl.gov
 tel:   865-241-3937
 fax:   865-572-0680
*/

#ifndef MADNESS_MRA_KEY_H__INCLUDED
#define MADNESS_MRA_KEY_H__INCLUDED

/// \file key.h
/// \brief Multidimension Key for MRA tree and associated iterators

#include <vector>
#include <madness/mra/power.h>
#include <madness/world/vector.h>
#include <madness/world/binary_fstream_archive.h>
#include <madness/world/worldhash.h>
#include <stdint.h>

namespace madness {

    //     // this has problems when nproc is a large-ish power of 2 such as 256
    //     // and leads to bad data distribution.
    //     static inline unsigned int sdbm(int n, const unsigned char* c, unsigned int sum=0) {
    //         for (int i=0; i<n; ++i) sum = c[i] + (sum << 6) + (sum << 16) - sum;
    //         return sum;
    //     }

    typedef int64_t Translation;
    typedef int Level;

    template<std::size_t NDIM>
    class KeyChildIterator;

    template<std::size_t NDIM>
    class KeyDescendantIterator;

    template<std::size_t NDIM>
    class KeyDescendantZOrderIterator;

    /// Key is the index for a node of the 2^NDIM-tree

    /// See KeyChildIterator for facile generation of children,
    /// and foreach_child(parent,op) for facile applicaiton of operators
    /// to child keys.
    template<std::size_t NDIM>
    class Key {
        friend class KeyChildIterator<NDIM>;
        friend class KeyDescendantIterator<NDIM>;
        friend class KeyDescendantZOrderIterator<NDIM>;
    private:
        Level n;
        Vector<Translation, NDIM> l;
        hashT hashval;


    public:
        /// Default constructor makes an \em uninitialized key
        Key() {}

        /// Constructor with given n, l
        Key(Level n, const Vector<Translation, NDIM>& l) : n(n), l(l)
        {
            rehash();
        }

        /// Constructor with given n and l=0
        Key(int n) : n(n), l(0)
        {
            rehash();
        }

        // /// easy constructor ... UGH !!!!!!!!!!!!!!!!!!!!!!
        // Key(const int n, const int l0, const int l1, const int l2) : n(n) {
        //     MADNESS_ASSERT(NDIM==3);
        //     l=Vector<Translation, NDIM>(0);
        //     l[0]=l0; l[1]=l1; l[2]=l2;
        //     rehash();
        // }

        /// Returns an invalid key
        static Key<NDIM>  invalid() {
            return Key<NDIM> (-1);
        }

        /// Checks if a key is invalid
        bool is_invalid() const {
            return n == -1;
        }

        /// Checks if a key is valid
        bool is_valid() const {
            return n != -1;
        }

        /// Equality test
        bool  operator==(const Key& other) const {
            if (hashval != other.hashval) return false;
            if (n != other.n) return false;
            for (unsigned int i=0; i<NDIM; i++)
                if (l[i] != other.l[i]) return false;
            return true; // everything is equal
        }

        bool operator!=(const Key& other) const {
            return !(*this == other);
        }

        /// Comparison operator less than to enable storage in STL map
        bool operator<(const Key& other) const {
            if (hashval < other.hashval) return true;
            if (hashval > other.hashval) return false;

            if (n < other.n) return true;
            if (n > other.n) return false;

            for (unsigned int i=0; i<NDIM; i++) {
                if (l[i] < other.l[i]) return true;
                if (l[i] > other.l[i]) return false;
            }

            return false; // everything is equal
        }

        inline hashT
        hash() const {
            return hashval;
        }

        // template<typename Archive>
        // inline void
        // serialize(Archive& ar) {
        //     ar & archive::wrap((unsigned char*) this, sizeof(*this));
        // }

        Level
        level() const {
            return n;
        }

        const Vector<Translation, NDIM>&
        translation() const {
            return l;
        }

        uint64_t
        distsq() const {
            uint64_t dist = 0;
            for (std::size_t d = 0; d < NDIM; ++d) {
                dist += l[d] * l[d];
            }
            return dist;
        }

        /// Returns the key of the parent

        /// Default is the immediate parent (generation=1).  To get
        /// the grandparent use generation=2, and similarly for
        /// great-grandparents.
        ///
        /// !! If there is no such parent it quietly returns the
        /// closest match (which may be self if this is the top of the
        /// tree).
        Key
        parent(int generation = 1) const {
            Vector<Translation, NDIM> pl;
            if (generation > n)
                generation = n;
            for (std::size_t i = 0; i < NDIM; ++i)
                pl[i] = l[i] >> generation;
            return Key(n - generation, pl);
        }


        bool
        is_child_of(const Key& key) const {
            if (this->n < key.n) {
                return false; // I can't be child of something lower on the tree
            }
            else if (this->n == key.n) {
                return (*this == key); // I am child of myself
            }
            else {
                Level dn = this->n - key.n;
                Key mama = this->parent(dn);
                return (mama == key);
            }
        }


        bool
        is_parent_of(const Key& key) const {
            return (key.is_child_of(*this));
        }

        /// Assuming keys are at the same level, returns true if displaced by no more than 1 in any direction

        /// Assumes key and this are at the same level
        bool
        is_neighbor_of(const Key& key, const std::vector<bool>& bperiodic) const {
            Translation dist = 0;
            Translation TWON1 = (Translation(1)<<n) - 1;
            for (std::size_t i=0; i<NDIM; ++i)
            {
                Translation ll = std::abs(l[i] - key.l[i]);
                if (bperiodic[i] && ll==TWON1) ll=1;
                dist = std::max(dist, ll);
            }
            return (dist <= 1);
        }

        /// given a displacement, generate a neighbor key; ignore boundary conditions and disp's level

        /// @param[in]  disp    the displacement
        /// @return     a new key
        Key neighbor(const Key<NDIM>& disp) const {
            Vector<Translation,NDIM> l = this->translation()+disp.translation();
            return Key(this->level(),l);
        }


        /// check if this MultiIndex contains point x, disregarding these two dimensions
        bool thisKeyContains(const Vector<double,NDIM>& x, const unsigned int& dim0,
                             const unsigned int& dim1) const {

            // it's sufficient if one single dimension is out
            bool contains=true;
            const double twotoN = std::pow(2.0,double(n));
            MADNESS_ASSERT(dim0<NDIM and dim1<NDIM);

            for (unsigned int i=0; i<NDIM; i++ ) {

                // check bounds
                MADNESS_ASSERT((x[i]>=0.0) and (x[i]<=1.0));

                // leave these two dimensions out
                if ((i==dim0) or (i==dim1)) continue;

                const int ll=int (x[i]*twotoN);
                if (not (l[i]==ll)) contains=false;
            }
            return contains;
        }

        /// break key into two low-dimensional keys
        template<std::size_t LDIM, std::size_t KDIM>
        void break_apart(Key<LDIM>& key1, Key<KDIM>& key2) const {

            // if LDIM==NDIM the 2nd key will be constructed empty
            MADNESS_ASSERT((LDIM+KDIM==NDIM) or (LDIM==NDIM));
            Vector<Translation, LDIM> l1;
            Vector<Translation, KDIM> l2;
            for (int i=0; i<static_cast<int>(LDIM); ++i) {
                l1[i]=l[i];
            }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
            for (size_t i=LDIM; i<NDIM; ++i) {
                l2[i-LDIM]=l[i];
            }
#pragma GCC diagnostic pop
	    
            key1=Key<LDIM>(n,l1);
            key2=Key<KDIM>(n,l2);
        }

        /// merge with other key (ie concatenate), use level of rhs, not of this
        template<std::size_t LDIM>
        Key<NDIM+LDIM> merge_with(const Key<LDIM>& rhs) const {
            Vector<Translation,NDIM+LDIM> t;
            for (int i=0; i<static_cast<int>(NDIM); ++i) t[i]     =this->l[i];
            for (int i=0; i<static_cast<int>(LDIM); ++i) t[NDIM+i]=rhs.translation()[i];
            return Key<NDIM+LDIM>(rhs.level(),t);
        }

        /// return if the other key is pointing in the same direction and is farther out

        /// unlike in distsq() the direction is taken into account, and other must be
        /// longer than this in each dimension
        /// @param[in]	other 	a key
        /// @return		if other is farther out
        bool is_farther_out_than(const Key<NDIM>& other) const {
            for (size_t i=0; i<NDIM; ++i) {
                if ((other.translation()[i]>0) and (other.translation()[i]>l[i])) return false;
                if ((other.translation()[i]<0) and (other.translation()[i]<l[i])) return false;
            }
            return true;
        }


        /// Recomputes hashval ... presently only done when reading from external storage
        void
        rehash() {
            //hashval = sdbm(sizeof(n)+sizeof(l), (unsigned char*)(&n));
            // default hash is still best

            hashval = hash_value(l);
            hash_combine(hashval, n);
        }
    };

    template<std::size_t NDIM>
    std::ostream&
    operator<<(std::ostream& s, const Key<NDIM>& key) {
        s << "(" << key.level() << "," << key.translation() << ")";
        return s;
    }

    /// given a source and a target, return the displacement in translation

    /// @param[in]  source  the source key
    /// @param[in]  target  the target key
    /// @return     disp    such that target = source + disp
    template<size_t NDIM>
    Key<NDIM> displacement(const Key<NDIM>& source, const Key<NDIM>& target) {
        MADNESS_ASSERT(source.level()==target.level());
        const Vector<Translation,NDIM> l = target.translation()-source.translation();
        return Key<NDIM>(source.level(),l);
    }



    /// Iterates in lexical order thru all children of a key

    /// Example usage:
    /// \code
    ///    for (KeyChildIterator<NDIM> it(key); it; ++it) print(it.key());
    /// \endcode
    template<std::size_t NDIM>
    class KeyChildIterator {
        Key<NDIM> parent;
        Key<NDIM> child;
        Vector<Translation, NDIM> p;
        bool finished;

    public:
        KeyChildIterator() :
                p(0), finished(true) {
        }

        KeyChildIterator(const Key<NDIM>& parent) :
                parent(parent), child(parent.n + 1, parent.l * 2), p(0),
                finished(false) {
        }

        /// Pre-increment of an iterator (i.e., ++it)
        KeyChildIterator&
        operator++() {
            if (finished)
                return *this;
            std::size_t i;
            for (i = 0; i < NDIM; ++i) {
                if (p[i] == 0) {
                    ++(p[i]);
                    ++(child.l[i]);
                    for (std::size_t j = 0; j < i; ++j) {
                        --(p[j]);
                        --(child.l[j]);
                    }
                    break;
                }
            }
            finished = (i == NDIM);
            child.rehash();
            return *this;
        }

        /// True if iterator is not at end
        operator bool() const {
            return !finished;
        }

        template<typename Archive>
        inline void
        serialize(Archive& ar) {
            ar & archive::wrap((unsigned char*) this, sizeof(*this));
        }

        /// Returns the key of the child
        inline const Key<NDIM>&
        key() const {
            return child;
        }
    };

    /// Iterates in lexical order thru all descendants at level key.level + levelDiff of a key

    /// Example usage:
    /// \code
    ///    for (KeyDescendantIterator<NDIM> it(key, 1); it; ++it) print(it.key());
    /// \endcode
    template<std::size_t NDIM>
    class KeyDescendantIterator {
        Key<NDIM> parent;
        Key<NDIM> child;
        Vector<madness::Translation, NDIM> p;
        Level levelDiff;
        madness::Translation maxTranslation;
        bool finished;

    public:
        KeyDescendantIterator() :
                p(0), levelDiff(0), finished(true) {
        }

        KeyDescendantIterator(const madness::Key<NDIM> &parent, madness::Level levelDiff) :
                parent(parent), child(parent.n + levelDiff, parent.l * (1u << static_cast<size_t>(levelDiff))), p(0),
                levelDiff(levelDiff), maxTranslation((1u << static_cast<size_t>(levelDiff)) - 1),
                finished(false) {
        }

        /// Pre-increment of an iterator (i.e., ++it)
        KeyDescendantIterator &operator++() {
            if (finished) { return *this; }

            std::size_t i;
            for (i = 0; i < NDIM; ++i) {
                if (p[i] < maxTranslation) {
                    ++(p[i]);
                    ++(child.l[i]);
                    for (std::size_t j = 0; j < i; ++j) {
                        p[j] -= maxTranslation;
                        child.l[j] -= maxTranslation;
                    }
                    break;
                }
            }

            finished = (i == NDIM);
            child.rehash();

            return *this;
        }

        /// True if iterator is not at end
        operator bool() const {
            return !finished;
        }

        template<typename Archive>
        inline void
        serialize(Archive& ar) {
            ar & archive::wrap((unsigned char*) this, sizeof(*this));
        }

        /// Returns the key of the child
        inline const madness::Key<NDIM> &key() const {
            return child;
        }
    };

    /// Iterates in Z-order thru all descendants at level key.level + levelDiff of a key

    /// Example usage:
    /// \code
    ///    for (KeyDescendantIterator<NDIM> it(key, 1); it; ++it) print(it.key());
    /// \endcode
    template<std::size_t NDIM>
    class KeyDescendantZOrderIterator {
        Key<NDIM> parent;
        Key<NDIM> child;
        Vector<madness::Translation, NDIM> p;
        Level levelDiff;
        madness::Translation maxTranslation;
        size_t remainingIter;

    public:
        KeyDescendantZOrderIterator() :
                p(0), levelDiff(0), remainingIter(0) {
        }

        KeyDescendantZOrderIterator(const madness::Key<NDIM> &parent, madness::Level levelDiff) :
                parent(parent), child(parent.n + levelDiff, parent.l * (1u << static_cast<size_t>(levelDiff))), p(0),
                levelDiff(levelDiff), maxTranslation((1u << static_cast<size_t>(levelDiff)) - 1),
                remainingIter(1u << (static_cast<size_t>(levelDiff) * NDIM)) {
        }

        /// Pre-increment of an iterator (i.e., ++it)
        KeyDescendantZOrderIterator &operator++() {
            if (remainingIter == 0) { return *this; }

            remainingIter--;

            size_t carry = 0x1;
            do {
                for (size_t i = 0; i < NDIM; i++) {
                    /* cary, x = x & cary, x ^ carry */
                    size_t tmpCarry = carry;
                    carry = p[i] & carry;
                    p[i] ^= tmpCarry;
                    child.l[i] ^= tmpCarry;
                }

                carry <<= 1u;
            } while (carry != 0);


            child.rehash();

            return *this;
        }

        /// True if iterator is not at end
        operator bool() const {
            return remainingIter > 0;
        }

        template<typename Archive>
        inline void
        serialize(Archive& ar) {
            ar & archive::wrap((unsigned char*) this, sizeof(*this));
        }

        /// Returns the key of the child
        inline const madness::Key<NDIM> &key() const {
            return child;
        }
    };


    /// Applies op(key) to each child key of parent
    template<std::size_t NDIM, typename opT>
    inline void
    foreach_child(const Key<NDIM>& parent, opT& op) {
        for (KeyChildIterator<NDIM>
                     it(parent); it; ++it)
            op(it.key());
    }

    /// Applies member function of obj to each child key of parent
    template<std::size_t NDIM, typename objT>
    inline void
    foreach_child(const Key<NDIM>& parent, objT* obj, void
    (objT::*memfun)(const Key<NDIM>&)) {
        for (KeyChildIterator<NDIM>
                     it(parent); it; ++it)
            (obj ->* memfun)(it.key());
    }

    namespace archive {

        // For efficiency serialize opaque so is just one memcpy, but
        // when reading from external storage rehash() so that we
        // can read data even if hash algorithm/function has changed.

        template <class Archive, std::size_t NDIM>
        struct ArchiveLoadImpl< Archive, Key<NDIM> > {
            static void load(const Archive& ar, Key<NDIM>& t) {
                ar & archive::wrap((unsigned char*) &t, sizeof(t));
            }
        };

        template <std::size_t NDIM>
        struct ArchiveLoadImpl< BinaryFstreamInputArchive, Key<NDIM> > {
            static void load(const BinaryFstreamInputArchive& ar, Key<NDIM>& t) {
                ar & archive::wrap((unsigned char*) &t, sizeof(t));
                t.rehash(); // <<<<<<<<<< This is the point
            }
        };

        template <class Archive, std::size_t NDIM>
        struct ArchiveStoreImpl< Archive, Key<NDIM> > {
            static void store(const Archive& ar, const Key<NDIM>& t) {
                ar & archive::wrap((unsigned char*) &t, sizeof(t));
            }
        };
    }

}

#endif // MADNESS_MRA_KEY_H__INCLUDED

