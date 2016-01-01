#include "directory.h"
#include "dirent.h"
#include <sys/types.h>
#include <sys/stat.h>
#include "string.h"

char* strupr(char *s)
{
    char *out = s;
    for(;*s;++s)
        *s = toupper(*s);
    return out;
}

std::string strMakeUp(std::string &res, std::string const &src)
{
    res = src;
    strupr(static_cast<char *>(&res[0]));
    return res;
}

void strMakeUp(std::string &res)
{
    strupr(static_cast<char *>(&res[0]));
}

int strStr(std::string const &sub, std::string const &main)
{
    const char *pos;
    size_t  result;
    size_t  len = main.size();
    pos = strstr( main.c_str(), sub.c_str() );
    result = pos - main.c_str();
    if ((result>=0)&&(result<len))
        return (int)result;
    else
        return -1;

}

bool maskcmp(std::string const &str,  std::string const &_mask)
{
  const char* s = &str.c_str()[0];
  const char* mask = &_mask.c_str()[0];
  const   char* cp=0;
  const   char* mp=0;
  for ( ; *s && *mask!='*'; mask++ ,s++ )
      if ( *mask != *s && *mask!='?')
          return 0;
  for (;;)
  {
    if (!*s)
    {
        while (*mask=='*')
            mask++;
        return ! *mask;
    }
    if (*mask=='*')
    {
        if (!*++mask) return 1;
        mp=mask;
        cp=s+1;
        continue;
    }
    if (*mask==*s||*mask=='?')
    {
        mask++, s++;
        continue;
    }
    mask=mp;
    s=cp++;
    }
}

bool maskicmp(std::string const &str, std::string const &_mask)
{
    std::string STR;
    std::string _MASK;
    strMakeUp(STR,str);
    strMakeUp(_MASK,_mask);
    return maskcmp(STR,_MASK);
}

Dir::Dir()
{
    m_open = false;
    m_size = 0;
    m_pos = m_filenames.begin();
}

Dir::~Dir(){}

int Dir::count()
{
    return m_size;
}

void Dir::close()
{
    m_open = false;
    m_filenames.clear();
    m_pos = m_filenames.begin();
    m_size = 0;
}

bool Dir::exist(std::string const &dirname) const
{
    DIR *ddir = opendir(dirname.c_str());
    if ( ddir == NULL )
    {
        return false;
    }
    closedir(ddir);
    return true;
}

void Dir::getFileList(std::string const &path,bool subdirs)
{
    DIR *ddir = opendir(path.c_str());
    if ( ddir == NULL )
    {
        close();
        return;
    }

    dirent *dir_file = NULL;
    while ( (dir_file = readdir(ddir) ) != NULL)
    {
        if( strcmp(dir_file->d_name, "." ) == 0 || strcmp(dir_file->d_name, ".." ) == 0 )
            continue;

        std::string str(path);
        if (path[path.length()-1] != '/')
            str.append("/");

        str.append(dir_file->d_name);

        struct stat buf;
        if (lstat(str.c_str(), &buf) != 0)
                    continue;

        if (S_ISREG(buf.st_mode))
        {
            std::string tmp;
            m_size++;

            if (str[0]=='.')
            {
                tmp.append(str.begin()+2,str.end());
            }
            else
            {
                tmp.append(str);
            }

            m_filenames.push_back(tmp);
        }
        else if (S_ISDIR(buf.st_mode) && subdirs)
        {
            getFileList(str.c_str(),subdirs);
        }
    }
    closedir(ddir);
}

bool Dir::open(std::string const &dirname, bool subdirs)
{
    std::string path;

    if (dirname == "" || dirname == " ")
    {
        path.append(".");
    }
    else
    {
        path.append(dirname);
        std::replace(path.begin(),path.end(),'\\','/');
    }

    close();
    if (exist(path))
    {
        getFileList(path.c_str(),subdirs);
    }
    m_open = true;

    return true;
}

bool Dir::find(std::string const &filename)
{
    for (auto i = m_filenames.begin(); i != m_filenames.end(); i++)
    {
        if (maskcmp((*i),filename))
            return true;
    }
    return false;

}

const std::list<std::string>& Dir::getFileList()
{
    return m_filenames;
}

bool Dir::isOpen()
{
    return m_open;
}
