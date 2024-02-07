package nann

// Fixed8 represents an extremely low precision floating point number used in the nann architecture.
//
// The Fixed8 is represented as follows:
//
//	0 0000 000
//	S IIII DDD
//
// Where S is the sign bit, I is the 5 bit integer part, and D is the 1 bit decimal part.
// The representable values are between -16.000 and +15.875.
// The smallest infinitesimal values are -0.125 and +0.125.
type Fixed8 int8

// Float32 returns a float32 representation of this fixed value.
func (x Fixed8) Float32() float32 { return float32(x) / 8 }
