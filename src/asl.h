#include <vector>
#include <asl.h>

class ASL
{ 
  public:
    static void initialize() ;
    static void finalize() ;
    static int  getRandom(size_t, float*) ; 

  private:
    ASL() ;
    ~ASL() ;
    static asl_random_t* rnd ;
} ; 


