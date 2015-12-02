#pragma once
#include "BezierPatch.h"
#include "Nurbs.h"
#include "bothInclude.h"
#include "AABB.h"
uint64_t const modelSignature = 0xBC00FFAAAAFF00CB;

#pragma pack(push, 1)
struct ModelHeader {
    uint64_t signature;
    uint64_t patchesCount;
    uint8_t  isNurbs;
    uint64_t bezierPatchesCount;
};

struct NurbsHeader {
    uint8_t n;
    uint8_t m;
    uint8_t order_u;
    uint8_t order_v;
};

struct BezierHeader {
    uint8_t n;
    uint8_t m;
};
#pragma pack(pop)

class Model {
public:
    Model() {
        m_model = nullptr;
    }
    ~Model() {
        if (m_model) {
            delete [] m_model;
            m_model = nullptr;
        }
    }

    static bool save(std::string const &path, NURBS const *model, uint64_t size);
    static bool save(std::string const &path, BezierPatch const *model, uint64_t size);


    bool load(std::string const &path);

    BezierPatch const *getPatchesPtr() const {
        return m_model;
    }

    uint64_t getSize() const {
        return m_size;
    }

    void clear() {
        if (m_model) {
            delete [] m_model;
            m_size = 0;
        }
    }
    AABB box;

private:
    bool loadBezier(FILE *file, ModelHeader &header);
    bool loadNurbs(FILE *file, ModelHeader &header);

    BezierPatch *m_model;
    uint64_t m_size;
};
