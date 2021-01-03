/**
 *
 */
package org.janelia.saalfeldlab.transformhelpers;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.Callable;
import java.util.stream.Stream;

import org.janelia.saalfeldlab.n5.hdf5.N5HDF5Reader;
import org.janelia.saalfeldlab.n5.imglib2.N5DisplacementField;

import ch.systemsx.cisd.hdf5.HDF5Factory;
import ch.systemsx.cisd.hdf5.IHDF5Reader;
import net.imglib2.realtransform.RealTransform;
import picocli.CommandLine;
import picocli.CommandLine.Option;

/**
 * @author Stephan Saalfeld &lt;saalfelds@janelia.hhmi.org&gt;
 * @author John Bogovic &lt;saalfelds@janelia.hhmi.org&gt;
 *
 */
public class TransformCoordinates implements Callable<Void> {

	@Option(names = {"-t", "--transform"}, required = true, description = "deformation field filename, e.g. JRC2018F_FCWB_transform_quant16.h5")
	private String transformFile;

	@Option(names = {"-c", "--coordinates"}, required = true, description = "coordinates filename, e.g. tALT.fafb.jrc2018.txt")
	private String coordinatesFile;

	@Option(names = {"-i", "--inverse"}, required = false)
	private boolean inverse;

	@Option(names = {"-l", "--level"}, required = false, description = "level [0-n] of multiresolution h5 registration."
			+ "If not specified, will choose the finest level present.")
	private int level = -1;

	public static void main(final String... args) {

		CommandLine.call(new TransformCoordinates(), args);
	}

	@Override
	public Void call() throws Exception {

		final IHDF5Reader hdf5Reader = HDF5Factory.openForReading(transformFile);
		final N5HDF5Reader n5 = new N5HDF5Reader(hdf5Reader, new int[]{16, 16, 16});

		String levelString;
		if( level >= 0 )
			levelString = String.format( "/%d", level );
		else
			levelString = n5.exists( "/0" ) ? "/0" : "";

		String path = levelString + (inverse ? "/invdfield" : "/dfield");
		RealTransform transform = N5DisplacementField.open(
				n5,
				path,
				inverse);

		final double[] p = new double[3];
		final double[] q = new double[3];

		try (Stream<String> stream = Files.lines(Paths.get(coordinatesFile))) {

			stream.forEach(line -> {
				final String[] coordinates = line.split(",?\\s+");
				if (coordinates.length == 3) {
					p[0] = Double.parseDouble(coordinates[0]);
					p[1] = Double.parseDouble(coordinates[1]);
					p[2] = Double.parseDouble(coordinates[2]);

					transform.apply(p, q);

					System.out.println(q[0] + " " + q[1] + " " + q[2]);
				}
			});
		}

		return null;
	}
}
