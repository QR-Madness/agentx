/**
 * ParallaxBackground — Multi-layer parallax effect with scroll interaction
 */

import { motion, useScroll, useTransform } from 'framer-motion';

export function ParallaxBackground() {
  const { scrollY } = useScroll();

  // Different parallax speeds for depth effect
  const y1 = useTransform(scrollY, [0, 1000], [0, -50]);
  const y2 = useTransform(scrollY, [0, 1000], [0, -100]);
  const y3 = useTransform(scrollY, [0, 1000], [0, -150]);

  return (
    <div className="parallax-container">
      {/* Layer 1: Radial gradient orb */}
      <motion.div className="parallax-layer parallax-layer-1" style={{ y: y1 }}>
        <div className="parallax-orb parallax-orb-1" />
        <div className="parallax-orb parallax-orb-2" />
      </motion.div>

      {/* Layer 2: Floating particles */}
      <motion.div className="parallax-layer parallax-layer-2" style={{ y: y2 }}>
        <div className="parallax-particles" />
      </motion.div>

      {/* Layer 3: Grid pattern overlay */}
      <motion.div className="parallax-layer parallax-layer-3" style={{ y: y3 }}>
        <div className="parallax-grid" />
      </motion.div>
    </div>
  );
}
