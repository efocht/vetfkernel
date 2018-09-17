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

/*
 * Log level
 * 1: Initialization, once per execution
 * 2: Kernel begin and end
 * 3 or more: Details
 */

#ifndef NDEBUG
#define LOG(lvl) \
  if ((lvl) <= LogMessage::getMinLogLevel()) LogMessage()

#else
#define LOG(lvl) \
  if (false) LogMessage()
#endif

#endif
