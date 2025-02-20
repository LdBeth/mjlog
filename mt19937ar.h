#include <stdint.h>

class _MTRND {
private:
  uint32_t mt[624]; /* the array for the state vector  */
  int mti; /* mti==N+1 means mt[N] is not initialized */
public:
 _MTRND(): mti(624+1) {};
  void init_genrand(uint32_t s);
  void init_by_array(uint32_t init_key[], int key_length);
  uint32_t genrand_int32(void);
};
