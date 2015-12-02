#include "Мodel.h"
#include <cstdio>
#include "Log.h"
#include <vector>


bool Model::loadBezier(FILE *file, ModelHeader &header) {
    BezierHeader bheader;
    m_model = new BezierPatch[header.patchesCount];
    m_size = header.patchesCount;

    if (!m_model) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "Model", "loadBezier", "memory was not alloced %s", SourcePos());
        return false;
    }
    for (uint64_t it = 0; it < header.patchesCount; it++) {
        if (!fread(&bheader, sizeof(BezierHeader), 1, file)) {
            return false;
        }

        if (bheader.m != 4 && bheader.n != 4) {
            Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "Model", "loadBezier", "model with patches degree lower or upper 3 are not supported %s", SourcePos());
            return false;
        }

        if (fread(m_model[it].rowf, sizeof(float), 64, file) != 64) {
            delete [] m_model;
            m_model = nullptr;
            Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "Model", "loadBezier", "model was corrupt");
            return false;
        }
    }
    return true;
}

bool Model::loadNurbs(FILE *file, ModelHeader &header) {
    NurbsHeader nheader;
    if (m_model) {
        delete [] m_model;
        m_model = nullptr;
    }
    m_model = new BezierPatch[header.bezierPatchesCount];

    if (!m_model) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "Model", "loadNurbs", "memory was not alloced %s", SourcePos());
        return false;
    }

    m_size = header.bezierPatchesCount;

    uint64_t offset = 0;
    NURBS tmp;
    for (uint64_t it = 0; it < header.patchesCount; it++) {
        if (!fread(&nheader, sizeof(NurbsHeader), 1, file)) {
            goto clear;
        }
        if (nheader.order_u != 4 || nheader.order_v != 4) {
            goto clear;
        }

        uint64_t knotUSize = nheader.n + nheader.order_u;
        uint64_t knotVSize = nheader.m + nheader.order_v;

        tmp.order_u = nheader.order_u;
        tmp.order_v = nheader.order_v;
        tmp.m = nheader.m;
        tmp.n = nheader.n;

        tmp.u_knots.resize(knotUSize);
        tmp.v_knots.resize(knotVSize);
        if (std::fread(&tmp.u_knots[0], sizeof(float), knotUSize, file) != knotUSize) {
            goto clear;
        }

        if (std::fread(&tmp.v_knots[0], sizeof(float), knotVSize, file) != knotVSize) {
            goto clear;
        }

        tmp.points.resize(tmp.n);
        for (int i = 0; i < nheader.n; i++) {
            tmp.points[i].resize(nheader.m);
            for (int j = 0; j < nheader.m; j++) {
                if (!fread(&tmp.points[i][j], sizeof(glm::vec4), 1, file)) {
                    goto clear;
                }
            }
        }
        tmp.decomposeSurface(m_model + offset);
        offset += tmp.getBezierPatchesCount();
    }
    return true;

    clear:
    Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "Model", "loadNurbs", "model was corrupt or not supported");
    delete [] m_model;
    m_model = nullptr;
    return false;
}

bool Model::load(std::string const &path) {
    FILE *file = std::fopen(path.c_str(), "rb");
    if (!file) {
        return false;
    }
    ModelHeader header;

    if (!fread(&header, sizeof(ModelHeader), 1, file)) {
        std::fclose(file);
        return false;
    }

    if (header.signature != modelSignature) {
        std::fclose(file);
        return false;
    }

    bool result;
    if (header.isNurbs) {
        result = loadNurbs(file, header);
    } else {
        result = loadBezier(file, header);
    }

    if (result) {
        box.computeAABB(reinterpret_cast<glm::vec4*>(m_model), 16 * m_size);
    }

    std::fclose(file);
    return result;
}

bool Model::save(std::string const &path, NURBS const *model, uint64_t size) {
    ModelHeader header;
    header.isNurbs = 1;
    header.signature = modelSignature;
    header.patchesCount = size;
    header.bezierPatchesCount = 0;

    FILE *file = fopen(path.c_str(), "wb");
    if (!file) {
        return false;
    }

    for (size_t i = 0; i < size; i++) {
        header.bezierPatchesCount += model[i].getBezierPatchesCount();
    }

    if (!std::fwrite(&header, sizeof(ModelHeader), 1, file)) { return false; }
    NurbsHeader nurbsHeader;

    for (size_t i = 0; i < size; i++) {
        nurbsHeader.n = model[i].n;
        nurbsHeader.m = model[i].m;
        nurbsHeader.order_u = model[i].order_u;
        nurbsHeader.order_v = model[i].order_v;
        uint64_t knotUSize = nurbsHeader.n + nurbsHeader.order_u;
        uint64_t knotVSize = nurbsHeader.m + nurbsHeader.order_v;

        if (std::fwrite(&nurbsHeader, sizeof(NurbsHeader), 1, file) != 1) { return false; }
        if (std::fwrite(&model[i].u_knots[0], sizeof(float), knotUSize, file) != knotUSize) { return false; }
        if (std::fwrite(&model[i].v_knots[0], sizeof(float), knotVSize, file) != knotVSize) { return false; }

        for (int u = 0; u < model[i].n; u++) {
            for (int v = 0; v < model[i].m; v++) {
                if (!std::fwrite(&model[i].points[u][v], sizeof(glm::vec4), 1, file)) { return false; }
            }
        }
    }
    std::fclose(file);
    return true;
}

bool Model::save(std::string const &path, BezierPatch const *model, uint64_t size) {
    ModelHeader header;
    header.isNurbs = 0;
    header.signature = modelSignature;
    header.patchesCount = size;
    header.bezierPatchesCount = size;

    FILE *file = fopen(path.c_str(), "wb");
    if (!file) {
        return false;
    }
    BezierHeader bezierHeader;
    if (!std::fwrite(&header, sizeof(ModelHeader), 1, file)) { return false; }
    for (uint64_t i = 0; i < size; i++) {
        bezierHeader.n = 4;
        bezierHeader.m = 4;
        if (!std::fwrite(&bezierHeader, sizeof(BezierHeader), 1, file)) { return false; }
        if (std::fwrite(model[i].rowf, sizeof(float), 4 * 16, file) != (4 * 16)) { return false; }
    }
    fclose(file);
    return true;
}

