#ifndef LOG_H
#define LOG_H

#include <sstream>

static int init_min_log_level()
{
  if (const char* tmp = getenv("VE_LOG_LEVEL")) {
    return atoi(tmp);
  }
  return 0;
}

class LogMessage : public std::basic_ostringstream<char> {
  public:
    LogMessage() {}
    ~LogMessage() {
      fprintf(stderr, "%s\n", str().c_str());
    }

    static int getMinLogLevel() {
      static int min_log_level = init_min_log_level();
      return min_log_level;
    }
};

#define LOG(lvl) \
  if ((lvl) <= LogMessage::getMinLogLevel()) LogMessage()

#endif
