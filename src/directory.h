#pragma once
#include "bothInclude.h"

class Dir
{
public:
    Dir();
    virtual ~Dir();
    bool  open(std::string const &dirname, bool subdirs =true);
    void  close();
    bool  find(std::string const &filename);
    int   count();
    bool  isOpen();
    bool  exist(std::string const &dirname) const;
    const std::list<std::string> &getFileList();
private:
    void  getFileList(std::string const &path, bool subdirs);
    int m_size;
    std::list<std::string> m_filenames;
    std::list<std::string>::iterator m_pos;
    bool              m_open;
};

